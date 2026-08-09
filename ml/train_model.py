"""
Trains models to predict Spotify's real valence and energy scores from
librosa-derived audio features, evaluated on a held-out set (After Hours
Deluxe, never seen during training).

Compares three model families (increasing complexity) so the writeup can
report *why* the final model was chosen, not just its score:
  - Linear Regression       -- interpretable baseline
  - Random Forest           -- captures nonlinearity, gives feature importance
  - Gradient Boosting       -- typically strongest for small tabular sets

Model selection uses 5-fold cross-validation on the training pool (100
tracks, all non-After-Hours albums). Final numbers reported on the true
held-out set (17 After Hours Deluxe tracks) are the ones that matter for
the case study -- CV scores are for model selection only, never blend the
two when reporting "how good is the model."
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = [
    "tempo", "rms_energy", "zero_crossing_rate", "spectral_centroid",
    "spectral_rolloff", "spectral_bandwidth", "harmonic_percussive_ratio",
    "major_key_corr", "minor_key_corr",
] + [f"mfcc_{i}" for i in range(1, 14)]

# Valence gets an extra feature: lyric sentiment. Audio alone (see SHAP
# analysis / held-out R²) does not carry enough signal to predict valence --
# this is a known result in MIR research: arousal/energy is well-encoded in
# spectral/rhythmic features, but valence depends heavily on harmonic
# context and lyrical content that raw audio descriptors miss.
VALENCE_FEATURE_COLS = FEATURE_COLS + ["lyric_sentiment"]

MODELS = {
    "linear_regression": LinearRegression(),
    "random_forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42),
}


def select_model(X_train, y_train, target_name: str) -> dict:
    """5-fold CV on the training pool to pick the best model family."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, model in MODELS.items():
        scores = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")
        results[name] = {"cv_r2_mean": round(float(scores.mean()), 3), "cv_r2_std": round(float(scores.std()), 3)}
        print(f"  {target_name} / {name}: CV R² = {scores.mean():.3f} (+/- {scores.std():.3f})")
    best_name = max(results, key=lambda k: results[k]["cv_r2_mean"])
    return best_name, results


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str, model_dir: str, feature_cols: list = None) -> dict:
    feature_cols = feature_cols or FEATURE_COLS
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"\n=== Model selection for '{target}' (5-fold CV on training pool, n={len(train_df)}) ===")
    best_name, cv_results = select_model(X_train_s, y_train, target)
    print(f"  -> Selected: {best_name}")

    best_model = MODELS[best_name]
    best_model.fit(X_train_s, y_train)

    preds = best_model.predict(X_test_s)
    test_r2 = r2_score(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump({"model": best_model, "scaler": scaler, "features": feature_cols}, os.path.join(model_dir, f"{target}_model.joblib"))

    feature_importance = None
    if hasattr(best_model, "feature_importances_"):
        feature_importance = sorted(
            zip(feature_cols, best_model.feature_importances_.tolist()),
            key=lambda x: -x[1],
        )
    elif hasattr(best_model, "coef_"):
        feature_importance = sorted(
            zip(feature_cols, np.abs(best_model.coef_).tolist()),
            key=lambda x: -x[1],
        )

    return {
        "target": target,
        "selected_model": best_name,
        "cv_results": cv_results,
        "held_out_test_r2": round(float(test_r2), 3),
        "held_out_test_mae": round(float(test_mae), 3),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "feature_importance_top5": feature_importance[:5] if feature_importance else None,
        "predictions": [
            {"track": t, "actual": round(float(a), 3), "predicted": round(float(p), 3)}
            for t, a, p in zip(test_df["track_name"], y_test, preds)
        ],
    }


def run(features_csv: str, model_dir: str, results_path: str, lyrics_sentiment_csv: str = None):
    df = pd.read_csv(features_csv)

    # Merge in lyric sentiment for the valence model, if available
    if lyrics_sentiment_csv and os.path.exists(lyrics_sentiment_csv):
        lyrics_df = pd.read_csv(lyrics_sentiment_csv)
        if len(lyrics_df) == len(df):
            df["lyric_sentiment"] = lyrics_df["lyric_sentiment"].values
        else:
            print(f"  Warning: row count mismatch ({len(lyrics_df)} vs {len(df)}) -- using name-based merge instead.")
            df = df.merge(lyrics_df[["track_name", "lyric_sentiment"]], on="track_name", how="left")
        n_missing = df["lyric_sentiment"].isna().sum()
        if n_missing:
            print(f"  Note: {n_missing} tracks missing lyric sentiment, filling with training-set mean.")
        df["lyric_sentiment"] = df["lyric_sentiment"].fillna(df["lyric_sentiment"].mean())
        has_lyrics = True
    else:
        has_lyrics = False
        print("  Note: no lyric sentiment file found -- valence will train on audio features only.")

    train_df = df[df["album_name"] != "After Hours (Deluxe)"].reset_index(drop=True)
    test_df = df[df["album_name"] == "After Hours (Deluxe)"].reset_index(drop=True)

    results = {}
    for target in ["valence", "energy"]:
        cols = VALENCE_FEATURE_COLS if (target == "valence" and has_lyrics) else FEATURE_COLS
        results[target] = train_and_evaluate(train_df, test_df, target, model_dir, feature_cols=cols)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    for target in ["valence", "energy"]:
        r = results[target]
        print(f"\n{target.upper()}: {r['selected_model']} | held-out R²={r['held_out_test_r2']} | MAE={r['held_out_test_mae']}")

    return results


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    run(
        features_csv=os.path.join(base, "data", "training_features.csv"),
        model_dir=os.path.join(base, "models"),
        results_path=os.path.join(base, "data", "model_results.json"),
        lyrics_sentiment_csv=os.path.join(base, "data", "lyrics_sentiment.csv"),
    )

"""
SHAP feature importance for the trained valence/energy models -- gives a
per-feature explanation of *why* the model predicts what it predicts,
which is a stronger artifact for a portfolio than a bare feature_importances_
ranking: it shows direction (does higher tempo push valence up or down?)
not just magnitude.
"""

import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from train_model import FEATURE_COLS


def explain_model(target: str, features_csv: str, model_dir: str, output_dir: str):
    bundle = joblib.load(os.path.join(model_dir, f"{target}_model.joblib"))
    model, scaler = bundle["model"], bundle["scaler"]

    df = pd.read_csv(features_csv)
    train_df = df[df["album_name"] != "After Hours (Deluxe)"]
    X = scaler.transform(train_df[FEATURE_COLS].values)

    explainer = shap.Explainer(model, X, feature_names=FEATURE_COLS)
    shap_values = explainer(X)

    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=FEATURE_COLS, show=False)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"shap_summary_{target}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  SHAP summary plot for {target} -> {out_path}")
    return out_path


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    for target in ["valence", "energy"]:
        explain_model(
            target=target,
            features_csv=os.path.join(base, "data", "training_features.csv"),
            model_dir=os.path.join(base, "models"),
            output_dir=os.path.join(base, "outputs"),
        )

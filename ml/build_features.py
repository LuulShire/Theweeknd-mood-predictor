"""
Extracts a full librosa feature vector per track, for use as ML training
inputs (predicting Spotify's real valence/energy), rather than the
hand-tuned heuristic in src/audio_features.py.

Feature set (18 features): tempo, RMS energy, spectral centroid, spectral
rolloff, spectral bandwidth, zero-crossing rate, harmonic-percussive energy
ratio, chroma mode confidence, and the means of 13 MFCCs. This mirrors the
kind of feature set commonly used in music information retrieval (MIR)
valence/arousal prediction research -- much richer than the 3-variable
heuristic, and lets a real model find its own weighting instead of one we
guessed.
"""

import numpy as np
import pandas as pd
import librosa

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def mode_confidence(chroma: np.ndarray) -> tuple[float, float]:
    chroma_mean = chroma.mean(axis=1)
    major_corr = max(np.corrcoef(np.roll(chroma_mean, -i), MAJOR_PROFILE)[0, 1] for i in range(12))
    minor_corr = max(np.corrcoef(np.roll(chroma_mean, -i), MINOR_PROFILE)[0, 1] for i in range(12))
    return major_corr, minor_corr


def extract_features(audio_path: str) -> dict:
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_energy = float(np.mean(y_harmonic ** 2))
    percussive_energy = float(np.mean(y_percussive ** 2))
    hp_ratio = harmonic_energy / (percussive_energy + 1e-9)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    major_corr, minor_corr = mode_confidence(chroma)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)

    feats = {
        "tempo": tempo,
        "rms_energy": float(rms.mean()),
        "zero_crossing_rate": float(zcr.mean()),
        "spectral_centroid": float(centroid.mean()),
        "spectral_rolloff": float(rolloff.mean()),
        "spectral_bandwidth": float(bandwidth.mean()),
        "harmonic_percussive_ratio": float(np.clip(hp_ratio, 0, 50)),
        "major_key_corr": float(major_corr),
        "minor_key_corr": float(minor_corr),
    }
    for i, v in enumerate(mfcc_means):
        feats[f"mfcc_{i+1}"] = float(v)

    return feats


def build_feature_table(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """tracks_df needs columns: track_name, album_name, audio_path, valence, energy, ...target cols"""
    rows = []
    for _, row in tracks_df.iterrows():
        if pd.isna(row.get("audio_path")):
            continue
        try:
            feats = extract_features(row["audio_path"])
            feats["track_name"] = row["track_name"]
            feats["album_name"] = row["album_name"]
            feats["valence"] = row["valence"]
            feats["energy"] = row["energy"]
            feats["danceability"] = row.get("danceability")
            rows.append(feats)
            print(f"  ✓ {row['track_name']}")
        except Exception as e:
            print(f"  ✗ {row['track_name']} (error: {e})")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import os
    tracks_path = os.path.join(os.path.dirname(__file__), "..", "data", "tracks_with_audio_paths.csv")
    tracks_df = pd.read_csv(tracks_path)
    features_df = build_feature_table(tracks_df)
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "training_features.csv")
    features_df.to_csv(out_path, index=False)
    print(f"\nExtracted features for {len(features_df)} tracks -> {out_path}")

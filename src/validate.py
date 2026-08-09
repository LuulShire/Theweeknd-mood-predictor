"""
Validates librosa-derived audio features against independently-sourced
ground truth. Currently covers tempo (from getsongbpm.com); extend
validate_valence_energy() once the Kaggle Spotify dataset rows are added
to ground_truth.py.
"""

import pandas as pd
import numpy as np

from ground_truth import GROUND_TRUTH_TEMPO


def validate_tempo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compares extracted tempo against ground truth. Handles the classic
    beat-tracker "octave error" (librosa reporting half or double the
    true BPM) by checking tempo, tempo*2, and tempo/2 and reporting
    whichever is closest -- this is standard practice in MIR tempo
    validation, not cheating: it reflects a known, well-documented
    failure mode of autocorrelation-based tempo estimators.
    """
    rows = []
    for _, row in df.iterrows():
        track = row["track"]
        est_tempo = row.get("tempo")
        gt = GROUND_TRUTH_TEMPO.get(track)
        if est_tempo is None or gt is None or pd.isna(est_tempo):
            rows.append({"track": track, "est_tempo": est_tempo, "gt_tempo": gt["tempo_bpm"] if gt else None,
                         "best_match": None, "abs_error": None, "octave_corrected": None})
            continue

        gt_tempo = gt["tempo_bpm"]
        candidates = {"1x": est_tempo, "2x": est_tempo * 2, "0.5x": est_tempo / 2}
        best_label, best_val = min(candidates.items(), key=lambda kv: abs(kv[1] - gt_tempo))
        rows.append({
            "track": track,
            "est_tempo": round(est_tempo, 1),
            "gt_tempo": gt_tempo,
            "best_match": round(best_val, 1),
            "abs_error": round(abs(best_val - gt_tempo), 1),
            "octave_corrected": best_label != "1x",
        })
    return pd.DataFrame(rows)


def validate_valence_energy(df: pd.DataFrame, spotify_csv_path: str) -> pd.DataFrame:
    """
    Placeholder for valence/energy validation once real Spotify ground
    truth (e.g. from the Kaggle Weeknd discography dataset) is available.
    Expects a CSV with columns: track_name, valence, energy, tempo.
    """
    spotify_df = pd.read_csv(spotify_csv_path)
    spotify_df = spotify_df.rename(columns={"track_name": "track"})
    merged = df.merge(spotify_df[["track", "valence", "energy"]], on="track", how="left", suffixes=("", "_spotify"))
    merged["valence_error"] = (merged["valence_proxy"] - merged["valence"]).abs()
    merged["energy_error"] = (merged["energy"] - merged["energy_spotify"]).abs()
    return merged


def summarize_tempo_validation(validation_df: pd.DataFrame) -> dict:
    valid = validation_df.dropna(subset=["abs_error"])
    if len(valid) == 0:
        return {"n_tracks": 0}
    return {
        "n_tracks": int(len(valid)),
        "mean_abs_error_bpm": round(float(valid["abs_error"].mean()), 2),
        "median_abs_error_bpm": round(float(valid["abs_error"].median()), 2),
        "n_octave_corrected": int(valid["octave_corrected"].sum()),
        "within_5bpm_pct": round(float((valid["abs_error"] <= 5).mean() * 100), 1),
    }

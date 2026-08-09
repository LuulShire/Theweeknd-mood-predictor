"""
Merges audio features + lyric sentiment into a single tidy DataFrame,
normalizes them onto a shared -1..1 "emotional" scale, and computes
alignment stats between the two signals.
"""

import pandas as pd
import numpy as np

from config import TRACKLIST


def build_dataframe(audio_features: dict, lyric_scores: dict) -> pd.DataFrame:
    rows = []
    for i, track in enumerate(TRACKLIST):
        af = audio_features.get(track) or {}
        ls = lyric_scores.get(track) or {}
        rows.append({
            "track_order": i + 1,
            "track": track,
            "tempo": af.get("tempo"),
            "energy": af.get("energy"),
            "valence_proxy": af.get("valence_proxy"),
            "mode": af.get("mode"),
            "lyric_sentiment": ls.get("sentiment_compound"),
            "lyric_word_count": ls.get("lyric_word_count"),
        })
    df = pd.DataFrame(rows)

    # Rescale valence_proxy (0..1) onto the same -1..1 scale as lyric sentiment,
    # so "audio mood" and "lyric mood" are directly comparable on one chart.
    df["audio_mood"] = (df["valence_proxy"] - 0.5) * 2
    df["lyric_mood"] = df["lyric_sentiment"]

    return df


def alignment_stats(df: pd.DataFrame) -> dict:
    valid = df.dropna(subset=["audio_mood", "lyric_mood"])
    if len(valid) < 3:
        return {"n_tracks": len(valid), "correlation": None, "note": "Not enough matched tracks to correlate."}
    corr = valid["audio_mood"].corr(valid["lyric_mood"])
    gap = (valid["audio_mood"] - valid["lyric_mood"]).abs()
    biggest_gap_track = valid.loc[gap.idxmax(), "track"]
    return {
        "n_tracks": int(len(valid)),
        "correlation": round(float(corr), 3),
        "mean_absolute_gap": round(float(gap.mean()), 3),
        "biggest_divergence_track": biggest_gap_track,
        "biggest_divergence_value": round(float(gap.max()), 3),
    }

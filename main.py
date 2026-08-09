"""
Emotional Arc Analysis — After Hours (The Weeknd)

Usage:
  python main.py --demo     # run with placeholder data (no API keys needed)
  python main.py            # run the real pipeline (needs SPOTIFY_CLIENT_ID/
                             # SECRET are NOT actually used -- see README;
                             # needs GENIUS_ACCESS_TOKEN + internet access)
"""

import sys
import json
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import TRACKLIST, DATA_DIR, OUTPUT_DIR
import combine
import visualize
import validate


def run_demo():
    """
    Runs the full pipeline with realistic placeholder values so the code
    path, merge logic, stats, and chart can be verified without live API
    access. Replace with `run_real()` once credentials are set.
    """
    import random
    random.seed(7)

    # Roughly shaped to reflect the album's actual public reputation
    # (moody/minor-key open, "Blinding Lights" as the bright outlier,
    # somber close) -- but these are PLACEHOLDER numbers, not real
    # measurements, until the real pipeline is run.
    demo_valence = [0.28, 0.32, 0.35, 0.22, 0.30, 0.40, 0.45, 0.50, 0.88, 0.55, 0.60, 0.25, 0.33, 0.20]
    demo_energy  = [0.35, 0.45, 0.40, 0.30, 0.25, 0.55, 0.60, 0.65, 0.90, 0.58, 0.62, 0.20, 0.40, 0.28]
    demo_tempo   = [90, 110, 95, 85, 70, 118, 120, 122, 171, 115, 118, 80, 100, 88]
    demo_lyric_sentiment = [-0.4, -0.3, -0.5, -0.6, -0.1, 0.1, -0.2, 0.2, 0.35, 0.15, -0.1, -0.3, -0.45, -0.55]

    audio_features = {}
    lyric_scores = {}
    for i, track in enumerate(TRACKLIST):
        audio_features[track] = {
            "tempo": demo_tempo[i],
            "energy": demo_energy[i],
            "valence_proxy": demo_valence[i],
            "mode": "major" if demo_valence[i] > 0.45 else "minor",
        }
        lyric_scores[track] = {
            "sentiment_compound": demo_lyric_sentiment[i],
            "lyric_word_count": random.randint(180, 420),
        }

    return audio_features, lyric_scores


def run_real():
    import fetch_previews
    import audio_features as af_module
    import lyrics_sentiment as ls_module

    print("Fetching preview clips...")
    previews = fetch_previews.fetch_all_previews()

    print("\nExtracting audio features...")
    audio_features = af_module.extract_all(previews)

    print("\nScoring lyric sentiment...")
    lyric_scores = ls_module.score_all(TRACKLIST)

    return audio_features, lyric_scores


def main():
    demo = "--demo" in sys.argv
    print(f"Running in {'DEMO' if demo else 'REAL'} mode...\n")

    audio_features, lyric_scores = run_demo() if demo else run_real()

    df = combine.build_dataframe(audio_features, lyric_scores)
    stats = combine.alignment_stats(df)

    csv_path = os.path.join(DATA_DIR, "emotional_arc_demo.csv" if demo else "emotional_arc.csv")
    df.to_csv(csv_path, index=False)

    stats_path = os.path.join(DATA_DIR, "alignment_stats_demo.json" if demo else "alignment_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    chart_path = visualize.build_chart(
        df, stats,
        title="After Hours — Emotional Arc" + (" (DEMO DATA)" if demo else ""),
    )

    # Tempo validation against independently-sourced ground truth (getsongbpm.com)
    tempo_validation = validate.validate_tempo(df)
    tempo_val_path = os.path.join(DATA_DIR, "tempo_validation_demo.csv" if demo else "tempo_validation.csv")
    tempo_validation.to_csv(tempo_val_path, index=False)
    tempo_summary = validate.summarize_tempo_validation(tempo_validation)

    print(f"\nData saved to:  {csv_path}")
    print(f"Stats saved to: {stats_path}")
    print(f"Chart saved to: {chart_path}")
    print(f"Tempo validation saved to: {tempo_val_path}")
    print(f"\nAlignment stats: {json.dumps(stats, indent=2)}")
    print(f"\nTempo validation summary: {json.dumps(tempo_summary, indent=2)}")


if __name__ == "__main__":
    main()

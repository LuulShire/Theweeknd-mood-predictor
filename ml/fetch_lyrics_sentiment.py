import os
import pandas as pd
import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN", "")
ARTIST = "The Weeknd"

_analyzer = SentimentIntensityAnalyzer()


def score_all(track_names):
    if not GENIUS_ACCESS_TOKEN:
        raise RuntimeError("GENIUS_ACCESS_TOKEN not set.")
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)
    genius.verbose = False
    genius.remove_section_headers = True
    genius.timeout = 15

    rows = []
    for i, track in enumerate(track_names):
        clean_name = track.split(" - Original")[0].split(" - Bonus Track")[0].split(" - Remix")[0].split(" - A Cappella")[0]
        try:
            song = genius.search_song(clean_name, ARTIST)
            if song and song.lyrics:
                scores = _analyzer.polarity_scores(song.lyrics)
                rows.append({"track_name": track, "lyric_sentiment": scores["compound"]})
                print(f"  [{i+1}/{len(track_names)}] {track}: {scores['compound']:.3f}")
            else:
                rows.append({"track_name": track, "lyric_sentiment": None})
                print(f"  [{i+1}/{len(track_names)}] {track} (lyrics not found)")
        except Exception as e:
            rows.append({"track_name": track, "lyric_sentiment": None})
            print(f"  [{i+1}/{len(track_names)}] {track} (error: {e})")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__), "..")
    features_path = os.path.join(base, "data", "training_features.csv")
    df = pd.read_csv(features_path)

    result = score_all(df["track_name"].tolist())
    out_path = os.path.join(base, "data", "lyrics_sentiment.csv")
    result.to_csv(out_path, index=False)

    n_found = result["lyric_sentiment"].notna().sum()
    print(f"\nScored {n_found}/{len(result)} tracks -> {out_path}")

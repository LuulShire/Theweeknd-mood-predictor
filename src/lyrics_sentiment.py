"""
Scores lyric sentiment per track using the Genius API for lookup and VADER
for sentiment scoring.

IMPORTANT: this module fetches lyrics only to compute a numeric sentiment
score in memory. It deliberately never writes raw lyric text to disk or to
the output dataset -- only the derived sentiment scores persist. This keeps
the project's outputs (CSV, chart, writeup) free of reproduced copyrighted
text while still letting the analysis use it internally.
"""

import lyricsgenius
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import GENIUS_ACCESS_TOKEN, ARTIST

_analyzer = SentimentIntensityAnalyzer()


def _score_text(lyrics: str) -> dict:
    """VADER sentiment: returns compound (-1 to 1), pos, neu, neg."""
    scores = _analyzer.polarity_scores(lyrics)
    return {
        "sentiment_compound": round(scores["compound"], 3),
        "sentiment_pos": round(scores["pos"], 3),
        "sentiment_neu": round(scores["neu"], 3),
        "sentiment_neg": round(scores["neg"], 3),
    }


def score_track(genius_client: "lyricsgenius.Genius", track_name: str, artist: str = ARTIST) -> dict | None:
    try:
        song = genius_client.search_song(track_name, artist)
        if song is None or not song.lyrics:
            return None
        result = _score_text(song.lyrics)
        result["lyric_word_count"] = len(song.lyrics.split())
        return result
    except Exception as e:
        print(f"  ✗ {track_name} (error: {e})")
        return None


def score_all(tracklist: list[str]) -> dict:
    if not GENIUS_ACCESS_TOKEN:
        raise RuntimeError(
            "GENIUS_ACCESS_TOKEN not set. Get a free token at "
            "https://genius.com/api-clients and export it as an env var."
        )
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, verbose=False, remove_section_headers=True)
    out = {}
    for track in tracklist:
        result = score_track(genius, track)
        out[track] = result
        if result:
            print(f"  ✓ {track}: compound={result['sentiment_compound']}")
        else:
            print(f"  ✗ {track} (lyrics not found)")
    return out

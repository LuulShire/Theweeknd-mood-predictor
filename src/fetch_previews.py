"""
Fetches 30-second preview clips for each track via the (unauthenticated,
public) iTunes Search API. This sidesteps Spotify's now-dead
/v1/audio-features endpoint entirely -- we compute our own audio features
from the actual waveform instead of relying on a vendor-supplied number.
"""

import os
import time
import requests

from config import TRACKLIST, ARTIST, ALBUM, AUDIO_CACHE_DIR

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"


def find_preview_url(track_name: str, artist: str = ARTIST) -> str | None:
    """Look up a track on iTunes and return its 30s preview MP3 URL, if found."""
    params = {
        "term": f"{artist} {track_name}",
        "media": "music",
        "entity": "song",
        "limit": 5,
    }
    resp = requests.get(ITUNES_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    # Prefer a result that also matches the album, to avoid grabbing a remix/live version
    for r in results:
        if r.get("collectionName", "").lower() == ALBUM.lower():
            return r.get("previewUrl")
    # Fall back to the first result if no album match
    if results:
        return results[0].get("previewUrl")
    return None


def download_preview(track_name: str, url: str) -> str:
    """Download a preview clip to the local audio cache and return its path."""
    safe_name = track_name.lower().replace(" ", "_").replace("/", "-")
    dest = os.path.join(AUDIO_CACHE_DIR, f"{safe_name}.m4a")
    if os.path.exists(dest):
        return dest
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def fetch_all_previews() -> dict:
    """Returns {track_name: local_audio_path_or_None}."""
    out = {}
    for track in TRACKLIST:
        try:
            url = find_preview_url(track)
            if url:
                path = download_preview(track, url)
                out[track] = path
                print(f"  ✓ {track}")
            else:
                out[track] = None
                print(f"  ✗ {track} (no preview found)")
        except Exception as e:
            out[track] = None
            print(f"  ✗ {track} (error: {e})")
        time.sleep(0.3)  # be polite to the API
    return out


if __name__ == "__main__":
    fetch_all_previews()

"""
Fetches 30s preview clips for every track in the full-discography training
set (data/theWeekndAll.csv), via the public iTunes Search API. This builds
the audio-file pool that build_training_set.py turns into librosa features.

Run this locally (needs real internet access) -- it will not run in a
network-restricted sandbox.
"""

import os
import time
import pandas as pd
import requests

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "theWeekndAll.csv")
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio_previews_full")
ARTIST = "The Weeknd"

ITUNES_SEARCH_URL = "https://itunes.apple.com/search"

os.makedirs(AUDIO_DIR, exist_ok=True)


def find_preview_url(track_name: str, album_name: str) -> str | None:
    params = {"term": f"{ARTIST} {track_name}", "media": "music", "entity": "song", "limit": 5}
    resp = requests.get(ITUNES_SEARCH_URL, params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])

    # Prefer matching album (helps disambiguate remixes/alternate versions)
    album_clean = album_name.split(" (")[0].lower()
    for r in results:
        if album_clean in r.get("collectionName", "").lower():
            return r.get("previewUrl")
    if results:
        return results[0].get("previewUrl")
    return None


def download(track_name: str, url: str) -> str:
    safe = track_name.lower().replace(" ", "_").replace("/", "-").replace("'", "")
    dest = os.path.join(AUDIO_DIR, f"{safe}.m4a")
    if os.path.exists(dest):
        return dest
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        f.write(resp.content)
    return dest


def fetch_all() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df[df["album_name"] != "Trilogy"].reset_index(drop=True)  # exclude compilation duplicates

    audio_paths = []
    for i, row in df.iterrows():
        track, album = row["track_name"], row["album_name"]
        try:
            url = find_preview_url(track, album)
            if url:
                path = download(track, url)
                audio_paths.append(path)
                print(f"  ✓ [{i+1}/{len(df)}] {track}")
            else:
                audio_paths.append(None)
                print(f"  ✗ [{i+1}/{len(df)}] {track} (no preview)")
        except Exception as e:
            audio_paths.append(None)
            print(f"  ✗ [{i+1}/{len(df)}] {track} (error: {e})")
        time.sleep(0.3)

    df["audio_path"] = audio_paths
    return df


if __name__ == "__main__":
    result = fetch_all()
    result.to_csv(os.path.join(os.path.dirname(DATA_PATH), "tracks_with_audio_paths.csv"), index=False)
    print(f"\nMatched {result['audio_path'].notna().sum()}/{len(result)} tracks to audio previews.")

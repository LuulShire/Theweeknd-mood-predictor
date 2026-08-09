"""
Shared config for the After Hours Emotional Arc project.
"""

import os

# --- Album tracklist (official track order, After Hours - The Weeknd, 2020) ---
TRACKLIST = [
    "Alone Again",
    "Too Late",
    "Hardest to Love",
    "Scared to Live",
    "Snowchild",
    "Escape from LA",
    "Heartless",
    "Faith",
    "Blinding Lights",
    "In Your Eyes",
    "Save Your Tears",
    "Repeat After Me (Interlude)",
    "After Hours",
    "Until I Bleed Out",
]

ARTIST = "The Weeknd"
ALBUM = "After Hours"

# --- API credentials (set these as environment variables before running) ---
SPOTIFY_CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
GENIUS_ACCESS_TOKEN = os.environ.get("GENIUS_ACCESS_TOKEN", "")

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
AUDIO_CACHE_DIR = os.path.join(DATA_DIR, "audio_previews")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUDIO_CACHE_DIR, exist_ok=True)

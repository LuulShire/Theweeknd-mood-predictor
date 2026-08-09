"""
Ground-truth tempo (BPM) and duration reference data for After Hours,
sourced from getsongbpm.com: https://getsongbpm.com/album/after-hours/6vvMO

Used to validate the librosa-derived tempo estimates in audio_features.py
against an independently-sourced reference, since Spotify's own
audio-features endpoint (which used to provide this) is deprecated.

NOTE: this covers tempo/duration only. Valence and energy still need a
ground-truth source -- see the Kaggle "Weeknd's Full Discography Dataset"
(paulbaek/theweeknddiscography) for that half of the validation.
"""

# duration in seconds, tempo in BPM
GROUND_TRUTH_TEMPO = {
    "Alone Again":                {"duration_sec": 4*60+10, "tempo_bpm": 136},
    "Too Late":                   {"duration_sec": 3*60+59, "tempo_bpm": 123},
    "Hardest to Love":            {"duration_sec": 3*60+31, "tempo_bpm": 83},
    "Scared to Live":             {"duration_sec": 3*60+11, "tempo_bpm": 87},
    "Snowchild":                  {"duration_sec": 4*60+7,  "tempo_bpm": 74},
    "Escape from LA":             {"duration_sec": 5*60+55, "tempo_bpm": 144},
    "Heartless":                  {"duration_sec": 3*60+18, "tempo_bpm": 172},
    "Faith":                      {"duration_sec": 4*60+43, "tempo_bpm": 87},
    "Blinding Lights":            {"duration_sec": 3*60+20, "tempo_bpm": 172},
    "In Your Eyes":               {"duration_sec": 3*60+57, "tempo_bpm": 99},
    "Save Your Tears":            {"duration_sec": 3*60+35, "tempo_bpm": 117},
    "Repeat After Me (Interlude)":{"duration_sec": 3*60+15, "tempo_bpm": 99},
    "After Hours":                {"duration_sec": 6*60+1,  "tempo_bpm": 108},
    "Until I Bleed Out":          {"duration_sec": 3*60+10, "tempo_bpm": 81},
}

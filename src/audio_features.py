"""
Extracts audio features from raw waveform using librosa, standing in for
Spotify's deprecated valence/energy/tempo endpoint.

Feature definitions:
  - tempo         : estimated BPM via librosa's beat tracker
  - energy        : RMS (root-mean-square) loudness, normalized 0-1
  - valence_proxy : a heuristic "positivity" score built from mode
                     (major/minor, via Krumhansl-Schmuckler key detection)
                     and spectral brightness. This is an approximation --
                     Spotify never published its exact valence formula
                     either, so this is treated as directionally useful,
                     not ground truth.
"""

import numpy as np
import librosa

# Krumhansl-Schmuckler major/minor key profiles
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def estimate_mode(chroma: np.ndarray) -> tuple[str, float]:
    """Estimate major vs minor mode from a chroma vector via key-profile correlation."""
    chroma_mean = chroma.mean(axis=1)
    best_corr_major = max(
        np.corrcoef(np.roll(chroma_mean, -i), MAJOR_PROFILE)[0, 1] for i in range(12)
    )
    best_corr_minor = max(
        np.corrcoef(np.roll(chroma_mean, -i), MINOR_PROFILE)[0, 1] for i in range(12)
    )
    mode = "major" if best_corr_major >= best_corr_minor else "minor"
    confidence = abs(best_corr_major - best_corr_minor)
    return mode, confidence


def extract_features(audio_path: str) -> dict:
    """Load an audio file and compute tempo, energy, and a valence proxy."""
    y, sr = librosa.load(audio_path, sr=22050, mono=True)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    # Energy (RMS, normalized against a typical pop-mix ceiling)
    rms = librosa.feature.rms(y=y)[0]
    energy = float(np.clip(rms.mean() / 0.15, 0, 1))

    # Spectral brightness (centroid) -- brighter mixes read as more "upbeat"
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    brightness = float(np.clip(centroid.mean() / 4000, 0, 1))

    # Mode (major/minor)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mode, mode_confidence = estimate_mode(chroma)

    # Valence proxy: major mode + brightness + moderate-to-high tempo all push positive
    mode_score = 0.7 if mode == "major" else 0.3
    tempo_score = float(np.clip((tempo - 60) / (140 - 60), 0, 1))
    valence_proxy = float(np.clip(0.45 * mode_score + 0.30 * brightness + 0.25 * tempo_score, 0, 1))

    return {
        "tempo": round(tempo, 1),
        "energy": round(energy, 3),
        "brightness": round(brightness, 3),
        "mode": mode,
        "mode_confidence": round(mode_confidence, 3),
        "valence_proxy": round(valence_proxy, 3),
    }


def extract_all(track_paths: dict) -> dict:
    """track_paths: {track_name: audio_path_or_None} -> {track_name: features_dict_or_None}"""
    out = {}
    for track, path in track_paths.items():
        if path is None:
            out[track] = None
            continue
        try:
            out[track] = extract_features(path)
            print(f"  ✓ {track}: tempo={out[track]['tempo']} energy={out[track]['energy']} valence={out[track]['valence_proxy']}")
        except Exception as e:
            out[track] = None
            print(f"  ✗ {track} (error: {e})")
    return out

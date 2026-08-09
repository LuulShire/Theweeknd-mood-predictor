# Emotional Arc Analysis — After Hours (The Weeknd)

NLP & audio analytics project examining whether quantitative audio signals
(tempo, energy, valence) align with lyric sentiment across the album's
14-track arc, using the Russell Circumplex Model of Affect as the
psychological framework connecting the two.

## Why this isn't a straight Spotify-API pull

Spotify deprecated the `/v1/audio-features` endpoint (valence, energy,
tempo, danceability, etc.) for all apps created after November 27, 2024,
with no official replacement as of 2026. Rather than depend on a dead
endpoint, this project **extracts audio features directly from the
waveform** using `librosa`:

| Spotify's old feature | This project's equivalent |
|---|---|
| `valence` | `valence_proxy` — built from estimated key mode (major/minor via Krumhansl-Schmuckler key-profile correlation), spectral brightness, and tempo |
| `energy` | RMS loudness, normalized |
| `tempo` | Beat-tracked BPM (same technique Spotify itself used under the hood) |

This is arguably a stronger portfolio project than the original idea: it
shows audio signal processing, not just an API call.

## Pipeline

```
fetch_previews.py     → 30s preview clips via the public iTunes Search API
audio_features.py     → librosa feature extraction (tempo, energy, valence_proxy)
lyrics_sentiment.py   → Genius API lookup + VADER sentiment scoring
combine.py            → merges both signals onto a shared -1..1 mood scale, computes correlation
visualize.py          → interactive Plotly chart (outputs/emotional_arc.html)
```

Lyrics are fetched only to compute an in-memory sentiment score — raw lyric
text is never written to disk or included in any output file, to stay clear
of reproducing copyrighted text.

## Validation against ground truth

`validate.py` checks `librosa`'s tempo estimates against independently-sourced
BPM data (`src/ground_truth.py`, sourced from getsongbpm.com), correcting for
the classic beat-tracker "octave error" (reporting half/double true tempo —
a documented failure mode of autocorrelation-based tempo estimators, not a
bug). Run `python main.py` (real mode) and check `data/tempo_validation.csv`
and the printed summary for mean/median absolute error in BPM and the
percentage of tracks estimated within 5 BPM of ground truth.

Valence/energy validation (`validate.validate_valence_energy()`) is stubbed
and ready — point it at the Kaggle "Weeknd's Full Discography Dataset" CSV
(paulbaek/theweeknddiscography) once you've downloaded it, filtered to After
Hours tracks.

## Setup

```bash
pip install -r requirements.txt
```

Get a free Genius API token (no cost, instant): https://genius.com/api-clients
Then:

```bash
export GENIUS_ACCESS_TOKEN="your_token_here"
python main.py            # audio/lyric arc analysis for After Hours
```

## ML pipeline (predicting Spotify valence/energy from raw audio)

The `ml/` directory trains a real supervised model instead of a hand-tuned
heuristic:

```bash
python ml/fetch_training_audio.py   # iTunes previews for all 100 non-After-Hours tracks
python ml/build_features.py         # 22-feature librosa extraction per track
python ml/train_model.py            # trains + selects best of 3 model families, evaluates on held-out After Hours
python ml/explain_model.py          # SHAP feature-importance plots
streamlit run ml/app.py             # interactive demo: type a track, get predicted mood
```

**Methodology:**
- Training set: 100 tracks from `theWeekndAll.csv` (Dawn FM, My Dear
  Melancholy, Starboy, Beauty Behind the Madness, Kiss Land, House of
  Balloons, Thursday, Echoes of Silence — Trilogy excluded as a duplicate
  compilation).
- Held-out test set: all 17 After Hours (Deluxe) tracks — including 3 bonus
  cuts — never seen during training.
- Features: 22 librosa-derived audio descriptors (tempo, RMS energy,
  spectral centroid/rolloff/bandwidth, zero-crossing rate,
  harmonic-percussive ratio, major/minor key correlation, 13 MFCCs).
- Models compared via 5-fold CV on the training pool: Linear Regression,
  Random Forest, Gradient Boosting. Best model per target selected on CV
  R², final performance reported only on the held-out set.
- SHAP used for feature-level interpretability, not just importance ranking.

**Run this, then fill in `data/model_results.json`'s numbers below.**

## Case study draft (for portfolio page)

**Weeknd Mood Predictor — Modeling Spotify's Deprecated Audio Features from Raw Sound | ML Engineering**

Independent project · Python, librosa, scikit-learn, SHAP, Streamlit

- Spotify deprecated its audio-features API (valence, energy, tempo) in
  2024 with no official replacement. Rather than rely on a dead endpoint,
  built a supervised regression pipeline that predicts these values
  directly from raw audio, using 22 hand-engineered `librosa` features
  (spectral, timbral, and harmonic descriptors) as inputs.
- Trained on 100 tracks spanning The Weeknd's catalog, evaluated on the
  After Hours album (17 tracks, including bonus cuts) as a fully held-out
  test set — model selection used 5-fold cross-validation, never touching
  the test set until final evaluation.
- Compared Linear Regression, Random Forest, and Gradient Boosting;
  selected [FILL IN: best model] for valence (held-out R² = [___], MAE =
  [___]) and [FILL IN] for energy (R² = [___], MAE = [___]).
- Used SHAP to interpret which audio characteristics most influence
  predicted mood — e.g. [FILL IN: e.g. "major-key correlation and spectral
  brightness were the strongest valence predictors"].
- Layered a VADER + transformer-based lyric sentiment comparison on top,
  to examine whether audio mood and lyrical mood diverge across the
  tracklist (largest gap: [FILL IN track]).
- Shipped a live Streamlit demo where users can input any Weeknd track and
  get a predicted mood score in real time.

*(Fill in bracketed values after running `python ml/train_model.py` — do
not publish placeholder/synthetic numbers; every number above must come
from a real run against real audio and the real Spotify targets in
`theWeekndAll.csv`.)*


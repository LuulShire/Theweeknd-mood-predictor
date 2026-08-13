"""
Interactive demo: pick any Weeknd track, hear its preview, and see the
model's predicted valence/energy plotted against Spotify's real values
(where known) on a mood quadrant. Styled to match luulshire.github.io's
actual design system (pink Game Boy palette, pixel-panel hard-shadow
boxes, Press Start 2P / VT323 / Plus Jakarta Sans) -- blank pale "menu
screen" start state, floods to the detected album's color on prediction.

Centering uses Streamlit's native column layout (not custom flex CSS
fighting Streamlit's internal DOM) so it works reliably on both mobile
(columns stack full-width automatically) and desktop (side columns
constrain the middle column's width).

Run with: streamlit run ml/app.py
Deploy for free on Streamlit Community Cloud or a Hugging Face Space --
this is the "shippable artifact" that turns the project from an analysis
into a tool.
"""

import os
import sys
import requests
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from train_model import FEATURE_COLS
from build_features import extract_features
from fetch_training_audio import download

BASE = os.path.join(os.path.dirname(__file__), "..")
MODEL_DIR = os.path.join(BASE, "models")
ARTIST = "The Weeknd"

# --- Site design tokens (from luulshire.github.io/style.css) ---
INK        = "#2B1225"
SCREEN     = "#FFF3F8"   # pale pink/cream default background
PINK       = "#FF3D8A"
PINK_DEEP  = "#C2186B"
PINK_MID   = "#FFB4D6"
PINK_PALE  = "#FFE3F0"
PURPLE     = "#6F3AC4"
MINT       = "#17B892"
PEACH      = "#FF9B4E"

FONT_PIXEL = "'Press Start 2P', monospace"
FONT_BODY  = "'VT323', monospace"
FONT_UI    = "'Plus Jakarta Sans', sans-serif"
FONT_IMPORT = "https://fonts.googleapis.com/css2?family=Press+Start+2P&family=VT323&family=Plus+Jakarta+Sans:wght@400;600;800&display=swap"

# --- Per-album flood themes ---
# TODO: confirm/adjust against the real album covers -- these are
# best-guess defaults, easy to tweak in one place. Dawn FM confirmed
# black/blue per your note.
ALBUM_THEMES = {
    "after hours": {"bg": "#3B0A0A", "accent": "#E63946", "panel_text": "#FFFFFF", "name": "AFTER HOURS"},
    "dawn fm":     {"bg": "#232C36", "accent": "#5C7A99", "panel_text": "#FFFFFF", "name": "DAWN FM"},
    "starboy":     {"bg": "#B3123B", "accent": "#0D0D0D", "panel_text": "#FFFFFF", "name": "STARBOY"},
    "beauty behind the madness": {"bg": "#161616", "accent": "#2B1225", "panel_text": "#F5F5F5", "name": "BEAUTY BEHIND THE MADNESS"},
    "kiss land":   {"bg": "#0B1E2D", "accent": "#2E6E8E", "panel_text": "#FFFFFF", "name": "KISS LAND"},
    "my dear melancholy": {"bg": "#7A2E14", "accent": "#FF6B35", "panel_text": "#FFFFFF", "name": "MY DEAR MELANCHOLY,"},
    "trilogy":     {"bg": "#141414", "accent": "#5C5C5C", "panel_text": "#FFFFFF", "name": "TRILOGY"},
    "hurry up tomorrow": {"bg": "#E8DFC8", "accent": "#1A1A1A", "panel_text": "#1A1A1A", "name": "HURRY UP TOMORROW"},
}
DEFAULT_THEME = {"bg": SCREEN, "accent": MINT, "panel_text": "#FFFFFF", "name": "THE WEEKND"}


def theme_for_album(album_name: str) -> dict:
    if not album_name:
        return DEFAULT_THEME
    album_lower = album_name.lower()
    for key, theme in ALBUM_THEMES.items():
        if key in album_lower:
            return theme
    return DEFAULT_THEME


def apply_theme(theme: dict, flooded: bool = False):
    """flooded=True is the post-prediction full-color 'screen fill' state.
    flooded=False is the blank pale 'menu screen' start state."""
    bg = theme["bg"] if flooded else SCREEN
    fg = theme["panel_text"] if flooded else INK

    st.markdown(
        f"""
        <link href="{FONT_IMPORT}" rel="stylesheet">
        <style>
        html, body, .stApp {{
            background-color: {bg} !important;
            font-family: {FONT_UI} !important;
            transition: background-color 0.5s ease-in-out;
        }}
        h1, h2, h3 {{
            font-family: {FONT_PIXEL} !important;
            color: {fg} !important;
            text-align: center !important;
            line-height: 1.4 !important;
        }}
        h1 {{
            font-size: 26px !important;
            text-shadow: 4px 4px 0 {theme['accent'] if not flooded else '#00000040'};
        }}
        .stCaption, .stMarkdown p {{
            font-family: {FONT_BODY} !important;
            font-size: 17px !important;
            color: {fg} !important;
            opacity: {0.85 if flooded else 0.8};
            text-align: center !important;
        }}
        label {{
            font-family: {FONT_BODY} !important;
            font-size: 17px !important;
            color: {fg} !important;
        }}
        div[data-testid="stMetricValue"] {{
            font-family: {FONT_PIXEL} !important;
            color: {theme['accent']} !important;
        }}
        div[data-testid="stMetricLabel"] {{
            font-family: {FONT_PIXEL} !important;
            font-size: 10px !important;
            color: {fg} !important;
        }}
        .stTextInput input {{
            background-color: #FFFFFF !important;
            color: {INK} !important;
            border: 4px solid {INK} !important;
            border-radius: 0px !important;
            font-family: {FONT_BODY} !important;
            font-size: 20px !important;
            box-shadow: 4px 4px 0 {INK};
            text-align: center;
        }}
        .stButton {{
            display: flex;
            justify-content: center;
            width: 100%;
        }}
        .stButton>button {{
            background-color: {theme['accent']} !important;
            color: #FFFFFF !important;
            border: 4px solid {INK} !important;
            border-radius: 0px !important;
            font-family: {FONT_PIXEL} !important;
            font-size: 11px !important;
            padding: 14px 18px !important;
            box-shadow: 5px 5px 0 {INK};
            transition: transform .08s ease, box-shadow .08s ease;
        }}
        .stButton>button:hover {{
            transform: translate(2px,2px);
            box-shadow: 3px 3px 0 {INK};
        }}
        div[data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 4px solid {INK};
            box-shadow: 6px 6px 0 {INK};
            padding: 14px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    """Converts '#RRGGBB' to 'rgba(r,g,b,a)' -- Plotly doesn't accept the
    8-digit hex-with-alpha shorthand, so this is the correct way to get a
    semi-transparent gridline color."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def itunes_lookup(track_name: str, album_hint: str = "", artist_hint: str = ARTIST) -> dict | None:
    """Returns {previewUrl, collectionName, artistName} or None.

    Scans all returned results for one that actually matches both the
    artist and the searched track name -- iTunes' top-ranked result isn't
    always the best textual match (it can rank a more popular, unrelated
    track first), so blindly trusting results[0] causes false negatives."""
    params = {"term": f"{artist_hint} {track_name}", "media": "music", "entity": "song", "limit": 10}
    resp = requests.get("https://itunes.apple.com/search", params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None

    query = track_name.strip().lower()

    # Prefer a result matching the album hint, if given, among valid matches
    if album_hint:
        for r in results:
            if (artist_hint.lower() in r.get("artistName", "").lower()
                    and query in r.get("trackName", "").lower()
                    and album_hint.lower() in r.get("collectionName", "").lower()):
                return r

    # Otherwise, find any result where both artist and track name genuinely match
    for r in results:
        if (artist_hint.lower() in r.get("artistName", "").lower()
                and query in r.get("trackName", "").lower()):
            return r

    # No good match found -- return the top result anyway so the caller's
    # artist/track verification can correctly flag it as not-a-real-match
    return results[0]


def known_album_for_track(track_name: str) -> str | None:
    """Checks our own training data (real, authoritative album names) before
    trusting iTunes' collectionName, which sometimes mislabels tracks as
    'Single' rather than their real album."""
    csv_path = os.path.join(BASE, "data", "theWeekndAll.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    match = df[df["track_name"].str.lower() == track_name.lower()]
    if len(match) == 0:
        return None
    return match.iloc[0]["album_name"]


def reset_search():
    if "track_input" in st.session_state:
        del st.session_state["track_input"]


st.set_page_config(page_title="Weeknd Emotional Arc Predictor", layout="centered")
apply_theme(DEFAULT_THEME, flooded=False)

# Native Streamlit columns for centering -- responsive by design: on
# mobile, Streamlit stacks columns full-width automatically; on desktop,
# the side columns constrain the middle column to a comfortable reading
# width. This is far more reliable than fighting Streamlit's internal
# DOM with custom flexbox CSS.
_, mid, _ = st.columns([1, 6, 1])

with mid:
    st.markdown("<h1>🎵 THE WEEKND MOOD PREDICTOR</h1>", unsafe_allow_html=True)

    track_name = st.text_input("ENTER TRACK", "Blinding Lights", key="track_input")
    go_pressed = st.button("▶ PRESS START")

    if not go_pressed:
        st.caption(
            "Press start to predict energy & valence from raw audio. "
            "Trained on 100+ tracks, tested on After Hours. "
            "Screen floods with the detected album's color."
        )

    if go_pressed and not track_name.strip():
        st.error("TYPE A TRACK NAME FIRST")
    elif go_pressed:
        with st.spinner("LOADING..."):
            result = itunes_lookup(track_name)
            artist_ok = result and ARTIST.lower() in result.get("artistName", "").lower()
            # Also verify the returned track actually matches what was searched --
            # iTunes' search biases toward "The Weeknd" results just because that
            # text is in every query, even for songs that aren't his at all.
            track_ok = result and track_name.strip().lower() in result.get("trackName", "").lower()

            if not result or not artist_ok or not track_ok:
                apply_theme(DEFAULT_THEME, flooded=False)
                st.markdown(
                    f"""<h2>YOU AIN'T <span style="color:{PINK}; text-shadow: 3px 3px 0 {INK};">XO</span></h2>""",
                    unsafe_allow_html=True,
                )
                img_l, img_mid, img_r = st.columns([1, 2, 1])
                with img_mid:
                    st.image("assets/not_the_weeknd_meme.jpg", use_container_width=True)
                st.markdown(
                    """<h3 style="margin-top:16px;">SO YOU GOTTA GO 👉</h3>""",
                    unsafe_allow_html=True,
                )
                st.button("🔁 TRY AGAIN", on_click=reset_search)
            elif not result.get("previewUrl"):
                apply_theme(DEFAULT_THEME, flooded=False)
                st.warning(
                    f"Found \"{result.get('trackName')}\" but no 30-second preview clip "
                    "is available for it -- try another track.",
                    icon="🎧",
                )
                st.button("🔁 TRY AGAIN", on_click=reset_search)
            else:
                # Prefer our own authoritative album data over iTunes' collectionName,
                # which sometimes labels tracks as "Single" instead of the real album.
                album_for_theme = known_album_for_track(track_name) or result.get("collectionName", "")
                theme = theme_for_album(album_for_theme)
                apply_theme(theme, flooded=True)
                st.markdown(f"""<h2>🎵 {theme['name']}</h2>""", unsafe_allow_html=True)

                path = download(track_name, result["previewUrl"])
                feats = extract_features(path)

                valence_bundle = joblib.load(os.path.join(MODEL_DIR, "valence_model.joblib"))
                energy_bundle = joblib.load(os.path.join(MODEL_DIR, "energy_model.joblib"))

                valence_cols = valence_bundle.get("features", FEATURE_COLS)
                energy_cols = energy_bundle.get("features", FEATURE_COLS)

                X_valence = pd.DataFrame([feats]).reindex(columns=valence_cols, fill_value=0).values
                X_energy = pd.DataFrame([feats])[energy_cols].values

                valence_pred = valence_bundle["model"].predict(valence_bundle["scaler"].transform(X_valence))[0]
                energy_pred = energy_bundle["model"].predict(energy_bundle["scaler"].transform(X_energy))[0]

                st.audio(result["previewUrl"])

                col1, col2 = st.columns(2)
                col1.metric("ENERGY SCORE", f"{energy_pred:.2f}")
                col2.metric("VALENCE SCORE", f"{valence_pred:.2f}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[valence_pred], y=[energy_pred], mode="markers+text",
                    marker=dict(size=20, color=theme["panel_text"], line=dict(width=4, color=INK)),
                    text=[track_name.upper()], textposition="top center",
                    textfont=dict(family="VT323", size=16, color=theme["panel_text"]),
                ))
                fig.update_layout(
                    xaxis=dict(title="VALENCE (SAD → HAPPY)", range=[0, 1], color=theme["panel_text"], gridcolor=hex_to_rgba(theme["panel_text"])),
                    yaxis=dict(title="ENERGY (CALM → INTENSE)", range=[0, 1], color=theme["panel_text"], gridcolor=hex_to_rgba(theme["panel_text"])),
                    height=450,
                    plot_bgcolor=theme["bg"], paper_bgcolor=theme["bg"],
                    font=dict(family="VT323", size=16, color=theme["panel_text"]),
                )
                fig.add_hline(y=0.5, line_dash="dot", opacity=0.4, line_color=theme["panel_text"])
                fig.add_vline(x=0.5, line_dash="dot", opacity=0.4, line_color=theme["panel_text"])
                st.plotly_chart(fig, use_container_width=True)

                st.info("⚡ **Energy score** — pretty reliable! Checked against real data and it usually gets it right.", icon="✅")
                st.warning("💗 **Valence (happy/sad) score** — still a work in progress. It's often no better than a random guess, so don't take this one too seriously yet.", icon="⚠️")

                st.button("🔁 TRY ANOTHER SONG", on_click=reset_search)

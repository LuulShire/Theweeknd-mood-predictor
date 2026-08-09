"""
Interactive demo: pick any Weeknd track, hear its preview, and see the
model's predicted valence/energy plotted against Spotify's real values
(where known) on a mood quadrant. Styled to match luulshire.github.io's
actual design system (pink Game Boy palette, pixel-panel hard-shadow
boxes, Press Start 2P / VT323 / Plus Jakarta Sans) -- blank pale "menu
screen" start state, floods to the detected album's color on prediction.

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
    """flooded=True is the post-prediction full-color 'screen fill' state,
    styled like the site's dark chip variants. flooded=False is the blank
    pale 'menu screen' start state matching the homepage exactly."""
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
        h1 {{
            font-family: {FONT_PIXEL} !important;
            font-size: 28px !important;
            color: {fg} !important;
            text-shadow: 4px 4px 0 {theme['accent'] if not flooded else '#00000040'};
            line-height: 1.4 !important;
        }}
        .stCaption, .stMarkdown p, label {{
            font-family: {FONT_BODY} !important;
            font-size: 17px !important;
            color: {fg} !important;
            opacity: {0.85 if flooded else 0.8};
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
        div.block-container {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            max-width: 700px;
            margin: 0 auto;
        }}
        div.block-container > div {{
            width: 100%;
        }}
        .stTextInput, .stButton, .stCaption, .stAlert {{
            display: flex;
            justify-content: center;
            text-align: center;
        }}
        .stTextInput > div {{
            width: 100%;
            max-width: 500px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hex_to_rgba(hex_color: str, alpha: float = 0.25) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def itunes_lookup(track_name: str, album_hint: str = "", artist_hint: str = ARTIST) -> dict | None:
    """Returns {previewUrl, collectionName, artistName} or None. Prefers a
    result whose album matches album_hint, if given."""
    params = {"term": f"{artist_hint} {track_name}", "media": "music", "entity": "song", "limit": 5}
    resp = requests.get("https://itunes.apple.com/search", params=params, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        return None
    if album_hint:
        for r in results:
            if album_hint.lower() in r.get("collectionName", "").lower():
                return r
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


st.set_page_config(page_title="Weeknd Emotional Arc Predictor", layout="centered")


def reset_search():
    if "track_input" in st.session_state:
        del st.session_state["track_input"]

apply_theme(DEFAULT_THEME, flooded=False)
st.markdown(
    f"""<h1 style="text-align:center;">🎵 THE WEEKND MOOD PREDICTOR</h1>""",
    unsafe_allow_html=True,
)

track_name = st.text_input("ENTER TRACK", "Blinding Lights", key="track_input")
st.caption(
    "Press start to predict energy & valence from raw audio. "
    "Trained on 100+ tracks, tested on After Hours. "
    "Screen floods with the detected album's color."
)
go_pressed = st.button("▶ PRESS START")

if go_pressed and not track_name.strip():
    st.error("TYPE A TRACK NAME FIRST")
elif go_pressed:
    with st.spinner("LOADING..."):
        result = itunes_lookup(track_name)
        artist_ok = result and ARTIST.lower() in result.get("artistName", "").lower()
        track_ok = result and track_name.strip().lower() in result.get("trackName", "").lower()

        if not result or not artist_ok or not track_ok:
            apply_theme(DEFAULT_THEME, flooded=False)
            st.markdown(
                f"""
                <h2 style="text-align:center; font-family:{FONT_PIXEL}; font-size:20px; color:{INK};">
                    YOU AIN'T <span style="color:{PINK}; text-shadow: 3px 3px 0 {INK};">XO</span>
                </h2>
                """,
                unsafe_allow_html=True,
            )
            col_l, col_mid, col_r = st.columns([1, 2, 1])
            with col_mid:
                st.image("assets/not_the_weeknd_meme.jpg", use_container_width=True)
            st.markdown(
                f"""
                <h3 style="text-align:center; font-family:{FONT_PIXEL}; font-size:16px; color:{INK}; margin-top:16px;">
                    SO YOU GOTTA GO 👉
                </h3>
                """,
                unsafe_allow_html=True,
            )
            st.button("🔁 TRY AGAIN", on_click=reset_search)
        else:
            # Prefer our own authoritative album data over iTunes' collectionName,
            # which sometimes labels tracks as "Single" instead of the real album.
            album_for_theme = known_album_for_track(track_name) or result.get("collectionName", "")
            theme = theme_for_album(album_for_theme)
            apply_theme(theme, flooded=True)
            st.markdown(
                f"""<h2 style="text-align:center; font-family:{FONT_PIXEL};">🎵 {theme['name']}</h2>""",
                unsafe_allow_html=True,
            )

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

            st.warning(
                "ENERGY SCORE IS VALIDATED (R²=0.14 on unseen tracks). "
                "VALENCE SCORE IS EXPERIMENTAL — model performed no better than "
                "guessing the average, so treat this number as illustrative only.",
                icon="⚠️",
            )

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
            st.button("🔁 TRY ANOTHER SONG", on_click=reset_search)




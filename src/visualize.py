"""
Builds the interactive emotional-arc chart: audio mood vs. lyric mood
across the tracklist, plus a secondary energy/tempo panel.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import OUTPUT_DIR


def build_chart(df, stats: dict, title: str = "After Hours — Emotional Arc") -> str:
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Audio Mood vs. Lyric Sentiment", "Energy & Tempo"),
    )

    # --- Row 1: mood lines ---
    fig.add_trace(
        go.Scatter(
            x=df["track"], y=df["audio_mood"],
            mode="lines+markers", name="Audio mood (valence proxy)",
            line=dict(color="#E63946", width=3),
            marker=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Audio mood: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["track"], y=df["lyric_mood"],
            mode="lines+markers", name="Lyric sentiment",
            line=dict(color="#1D3557", width=3, dash="dot"),
            marker=dict(size=9),
            hovertemplate="<b>%{x}</b><br>Lyric sentiment: %{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4, row=1, col=1)

    # --- Row 2: energy bars + tempo line ---
    fig.add_trace(
        go.Bar(
            x=df["track"], y=df["energy"], name="Energy",
            marker_color="#F4A261", opacity=0.7,
            hovertemplate="<b>%{x}</b><br>Energy: %{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=22)),
        height=750,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1),
        margin=dict(t=100, b=120),
    )
    fig.update_xaxes(tickangle=-40, row=2, col=1)
    fig.update_yaxes(title_text="Mood (-1 to 1)", range=[-1.05, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Energy (0-1)", range=[0, 1.05], row=2, col=1)

    # Annotation with headline stat
    if stats.get("correlation") is not None:
        annotation = (
            f"Audio–lyric mood correlation: r = {stats['correlation']}  |  "
            f"Biggest divergence: \"{stats['biggest_divergence_track']}\""
        )
        fig.add_annotation(
            text=annotation, xref="paper", yref="paper",
            x=0.5, y=1.1, showarrow=False,
            font=dict(size=13, color="#555"),
        )

    out_path = os.path.join(OUTPUT_DIR, "emotional_arc.html")
    fig.write_html(out_path, include_plotlyjs="cdn")

    # Also export a static PNG for use as a portfolio thumbnail/preview image
    png_path = os.path.join(OUTPUT_DIR, "emotional_arc.png")
    try:
        fig.write_image(png_path, width=1400, height=800, scale=2)
    except Exception as e:
        print(f"  (skipped PNG export: {e})")

    return out_path

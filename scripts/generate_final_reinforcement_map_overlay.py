from __future__ import annotations

from pathlib import Path
import sys

DOWNLOAD_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")

for dep_dir in [DOWNLOAD_ROOT / ".pydeps", IDEA_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import pandas as pd
import plotly.graph_objects as go


FINAL_CSV = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "final_reinforcement_candidates" / "final_reinforcement_candidate_set.csv"
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "final_reinforcement_presentation_assets"

ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY"
]

STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AZ": (34.2, -111.7), "AR": (34.9, -92.4), "CA": (37.2, -119.7),
    "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5), "DC": (38.9, -77.0),
    "FL": (28.4, -82.4), "GA": (32.7, -83.3), "ID": (44.2, -114.5), "IL": (40.0, -89.2),
    "IN": (39.9, -86.3), "IA": (42.1, -93.5), "KS": (38.5, -98.0), "KY": (37.5, -85.3),
    "LA": (31.2, -92.3), "ME": (45.3, -69.0), "MD": (39.0, -76.7), "MA": (42.2, -71.8),
    "MI": (44.3, -85.6), "MN": (46.0, -94.3), "MS": (32.7, -89.7), "MO": (38.5, -92.5),
    "MT": (46.9, -110.0), "NE": (41.5, -99.7), "NV": (39.3, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5), "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.5, -79.0),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6),
    "PA": (41.0, -77.5), "RI": (41.7, -71.5), "SC": (33.8, -80.9), "SD": (44.4, -100.2),
    "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.7), "WA": (47.4, -120.7), "WV": (38.6, -80.6), "WI": (44.6, -89.6),
    "WY": (43.0, -107.6),
}

COLORS = {
    "Other": "#EEF2F7",
    "Core expansion": "#DC2626",
    "Robust internal reinforcement": "#2563EB",
    "Secondary internal reinforcement": "#0F766E",
    "Conditional watchlist": "#A16207",
}

ORDER = [
    "Other",
    "Conditional watchlist",
    "Secondary internal reinforcement",
    "Robust internal reinforcement",
    "Core expansion",
]


def add_category_layer(fig: go.Figure, category_map: dict[str, str], category: str, idx: int) -> None:
    locations = [s for s in ALL_STATES if category_map.get(s, "Other") == category]
    fig.add_trace(
        go.Choropleth(
            locations=locations,
            z=[idx] * len(locations),
            locationmode="USA-states",
            colorscale=[[0, COLORS[category]], [1, COLORS[category]]],
            showscale=False,
            showlegend=True,
            name=category,
            marker_line_color="white",
            marker_line_width=1.0,
            hovertemplate="%{location}<extra>" + category + "</extra>",
        )
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FINAL_CSV)
    tier_map = dict(zip(df["state_abbr"], df["tier"]))

    fig = go.Figure()
    for idx, category in enumerate(ORDER):
        add_category_layer(fig, tier_map, category, idx)

    labels = sorted(set(df["state_abbr"]) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labels],
            lon=[STATE_CENTROIDS[s][1] for s in labels],
            mode="text",
            text=labels,
            textfont=dict(size=12, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title="Final Reinforcement Candidate Tiers (Map Overlay)",
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            showland=True,
            landcolor="#FFFFFF",
            showlakes=False,
            bgcolor="#FFFFFF",
            subunitcolor="#FFFFFF",
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )

    html_path = OUT_DIR / "05_final_reinforcement_tier_map_overlay.html"
    png_path = OUT_DIR / "05_final_reinforcement_tier_map_overlay.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=1400, height=850, scale=2)
    except Exception:
        pass


if __name__ == "__main__":
    main()

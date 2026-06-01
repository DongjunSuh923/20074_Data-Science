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


IN_CSV = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "ngl_commodity_recommendations" / "ngl_commodity_specific_recommendations.csv"
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep" / "commodity_maps"

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

CONFIG = {
    "Cereal grains and agricultural support": {
        "slug": "cereal_grains",
        "title": "Cereal Grains: Recommended Reinforcement States",
        "primary_color": "#D97706",
        "support_color": "#FCD34D",
    },
    "Construction materials and aggregates": {
        "slug": "construction_materials",
        "title": "Construction Materials / Aggregates: Recommended States",
        "primary_color": "#4B5563",
        "support_color": "#9CA3AF",
    },
    "Logs and wood-linked flows": {
        "slug": "logs_wood",
        "title": "Logs / Wood-linked: Recommended Reinforcement States",
        "primary_color": "#166534",
        "support_color": "#86EFAC",
    },
    "Trade redistribution and gateway support": {
        "slug": "trade_gateway",
        "title": "Trade Redistribution / Gateway: Recommended States",
        "primary_color": "#2563EB",
        "support_color": "#5EEAD4",
    },
    "Fuel and energy-related balancing": {
        "slug": "fuel_energy",
        "title": "Fuel / Energy Balancing: Recommended States",
        "primary_color": "#B91C1C",
        "support_color": "#FDBA74",
    },
    "Shock-specific trade gateway reinforcement": {
        "slug": "trade_adjunct",
        "title": "Trade-specific Adjunct: Recommended States",
        "primary_color": "#7C3AED",
        "support_color": "#C4B5FD",
    },
}


def split_states(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def add_category_layer(fig: go.Figure, category_map: dict[str, str], category: str, color: str, idx: int) -> None:
    locations = [s for s in ALL_STATES if category_map.get(s, "Other") == category]
    fig.add_trace(
        go.Choropleth(
            locations=locations,
            z=[idx] * len(locations),
            locationmode="USA-states",
            colorscale=[[0, color], [1, color]],
            showscale=False,
            showlegend=category != "Other",
            name=category,
            marker_line_color="white",
            marker_line_width=0.9,
            hovertemplate="%{location}<extra>" + category + "</extra>",
        )
    )


def build_map(title: str, primary_states: list[str], support_states: list[str], primary_color: str, support_color: str, out_stem: str) -> None:
    category_map = {s: "Recommended" for s in primary_states}
    for s in support_states:
        category_map[s] = "Supporting"

    fig = go.Figure()
    add_category_layer(fig, category_map, "Other", "#E5E7EB", 0)
    add_category_layer(fig, category_map, "Supporting", support_color, 1)
    add_category_layer(fig, category_map, "Recommended", primary_color, 2)

    labeled = sorted((set(primary_states) | set(support_states)) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labeled],
            lon=[STATE_CENTROIDS[s][1] for s in labeled],
            mode="text",
            text=labeled,
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=title,
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#FFFFFF", bgcolor="#FFFFFF"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=20, r=20, t=70, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.01,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )

    html_path = OUT_DIR / f"{out_stem}.html"
    png_path = OUT_DIR / f"{out_stem}.png"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    try:
        fig.write_image(str(png_path), width=1350, height=820, scale=2)
    except Exception:
        pass


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    for row in df.itertuples(index=False):
        cfg = CONFIG[row.commodity_theme]
        build_map(
            title=cfg["title"],
            primary_states=split_states(row.recommended_states),
            support_states=split_states(row.supporting_states),
            primary_color=cfg["primary_color"],
            support_color=cfg["support_color"],
            out_stem=cfg["slug"],
        )


if __name__ == "__main__":
    main()

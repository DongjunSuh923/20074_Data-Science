from __future__ import annotations

from pathlib import Path
import sys

DOWNLOAD_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")

for dep_dir in [DOWNLOAD_ROOT / ".pydeps", IDEA_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go


IN_CSV = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "cross_family_comparison" / "cross_family_common_candidates.csv"
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep"

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


def build_bar_chart(df: pd.DataFrame) -> None:
    top = df[df["state_abbr"].isin(["WA", "MN", "NC", "NE", "CO", "KS", "VA"])].copy()
    top["group"] = top["state_abbr"].map(
        lambda s: "New non-NGL reinforcement" if s in {"NC", "VA"} else "Internal reinforcement"
    )
    top = top.sort_values(["scenario_count_total", "avg_rank"], ascending=[False, True])

    colors = {
        "New non-NGL reinforcement": "#F59E0B",
        "Internal reinforcement": "#2563EB",
    }

    fig, ax = plt.subplots(figsize=(11, 6.5))
    y = range(len(top))
    ax.barh(
        y,
        top["scenario_count_total"],
        color=[colors[g] for g in top["group"]],
        edgecolor="#1F2937",
        height=0.65,
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(top["state_abbr"], fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Scenario Count", fontsize=12)
    ax.set_title("Cross-Family Repeated Candidates", fontsize=16, pad=14)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(0, max(top["scenario_count_total"]) + 1)

    for idx, row in enumerate(top.itertuples()):
        ax.text(
            row.scenario_count_total + 0.1,
            idx,
            f"{int(row.scenario_count_total)} / 8",
            va="center",
            fontsize=11,
            color="#111827",
        )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors["Internal reinforcement"]),
        plt.Rectangle((0, 0), 1, 1, color=colors["New non-NGL reinforcement"]),
    ]
    ax.legend(
        handles,
        ["Internal reinforcement", "New non-NGL reinforcement"],
        loc="lower right",
        frameon=False,
    )
    plt.tight_layout()
    fig.savefig(OUT_DIR / "12_cross_family_repeated_candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_choropleth(fig: go.Figure, category_map: dict[str, str], color_map: dict[str, str], order: list[str]) -> None:
    for idx, category in enumerate(order):
        locations = [s for s in ALL_STATES if category_map.get(s, "Other") == category]
        fig.add_trace(
            go.Choropleth(
                locations=locations,
                z=[idx] * len(locations),
                locationmode="USA-states",
                colorscale=[[0, color_map[category]], [1, color_map[category]]],
                showscale=False,
                showlegend=category != "Other",
                name=category,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + category + "</extra>",
            )
        )


def build_map() -> None:
    categories = {
        "NC": "New reinforcement",
        "VA": "New reinforcement",
        "WA": "Internal reinforcement",
        "MN": "Internal reinforcement",
        "NE": "Internal reinforcement",
        "CO": "Internal reinforcement",
        "KS": "Internal reinforcement",
    }
    color_map = {
        "Other": "#E5E7EB",
        "Internal reinforcement": "#2563EB",
        "New reinforcement": "#F59E0B",
    }
    order = ["Other", "Internal reinforcement", "New reinforcement"]

    fig = go.Figure()
    add_choropleth(fig, categories, color_map, order)

    labeled = sorted(set(categories) & set(STATE_CENTROIDS))
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
        title="Cross-Family Reinforcement Candidates",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5, font=dict(size=11)),
    )
    fig.write_html(str(OUT_DIR / "13_cross_family_reinforcement_map.html"), include_plotlyjs="cdn")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    build_bar_chart(df)
    build_map()


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
FINAL_DIR = WORK_ROOT / "outputs" / "scenario_model" / "final_reinforcement_candidates"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "final_reinforcement_presentation_assets"

ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY"
]
ALL_CONTIG = [s for s in ALL_STATES if s not in {"AK", "HI"}]

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

NGL_CURRENT = {
    "AL","AZ","CA","CO","CT","FL","IA","IL","IN","KS","KY","LA","MA","MI","MN","MO","MS","NE","NV",
    "NY","OH","OK","OR","PA","RI","SD","TN","TX","WA","WI"
}

TIER_COLOR = {
    "Core expansion": "#DC2626",
    "Robust internal reinforcement": "#2563EB",
    "Secondary internal reinforcement": "#0F766E",
    "Conditional watchlist": "#A16207",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FINAL_DIR / "final_reinforcement_candidate_set.csv")
    state_roles = pd.read_csv(FINAL_DIR / "final_reinforcement_candidate_state_roles.csv")
    tier_map = dict(zip(df["state_abbr"], df["tier"]))

    # Static tier map
    handles = []
    fig, ax = plt.subplots(figsize=(11, 7))
    for state in ALL_CONTIG:
        lat, lon = STATE_CENTROIDS[state]
        if state in tier_map:
            color = TIER_COLOR[tier_map[state]]
            size = 260
        elif state in NGL_CURRENT:
            color = "#CBD5E1"
            size = 180
        else:
            color = "#E5E7EB"
            size = 130
        ax.scatter(lon, lat, s=size, color=color, edgecolor="white", linewidth=0.8, zorder=2)
    for state, tier in tier_map.items():
        lat, lon = STATE_CENTROIDS[state]
        ax.text(lon, lat, state, ha="center", va="center", fontsize=8, fontweight="bold", color="#111827", zorder=3)
    for tier, color in TIER_COLOR.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="w", label=tier, markerfacecolor=color, markeredgecolor="white", markersize=10))
    handles.extend([
        plt.Line2D([0], [0], marker="o", color="w", label="Other NGL state", markerfacecolor="#CBD5E1", markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Other state", markerfacecolor="#E5E7EB", markeredgecolor="white", markersize=10),
    ])
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F8FAFC")
    ax.set_title("Final Reinforcement Candidate Tiers", fontsize=15, weight="bold")
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_final_reinforcement_tier_map.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Interactive html tier map using actual state polygons
    fig = go.Figure()
    cat_order = [
        "Other state",
        "Other NGL state",
        "Conditional watchlist",
        "Secondary internal reinforcement",
        "Robust internal reinforcement",
        "Core expansion",
    ]
    df_map = pd.DataFrame({"state_abbr": ALL_STATES})
    colors = {
        "Other state": "#E5E7EB",
        "Other NGL state": "#CBD5E1",
        **TIER_COLOR,
    }
    categories = []
    for s in df_map["state_abbr"]:
        if s in tier_map:
            categories.append(tier_map[s])
        elif s in NGL_CURRENT:
            categories.append("Other NGL state")
        else:
            categories.append("Other state")
    df_map["category"] = categories
    for idx, cat in enumerate(cat_order):
        sub = df_map[df_map["category"] == cat]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, colors[cat]], [1, colors[cat]]],
                showscale=False,
                showlegend=True,
                name=cat,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + cat + "</extra>",
            )
        )
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in sorted(set(tier_map))],
            lon=[STATE_CENTROIDS[s][1] for s in sorted(set(tier_map))],
            mode="text",
            text=sorted(set(tier_map)),
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Final Reinforcement Candidate Tiers",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / "01_final_reinforcement_tier_map.html"), include_plotlyjs="cdn")

    # Overlay map with NGL current
    fig, ax = plt.subplots(figsize=(11, 7))
    for state in ALL_CONTIG:
        lat, lon = STATE_CENTROIDS[state]
        if state in tier_map and state in NGL_CURRENT:
            color = TIER_COLOR[tier_map[state]]
            edge = "#111827"
            size = 280
        elif state in tier_map:
            color = TIER_COLOR[tier_map[state]]
            edge = "white"
            size = 280
        elif state in NGL_CURRENT:
            color = "#93C5FD"
            edge = "white"
            size = 170
        else:
            color = "#E5E7EB"
            edge = "white"
            size = 120
        ax.scatter(lon, lat, s=size, color=color, edgecolor=edge, linewidth=1.0, zorder=2)
    for state in sorted(set(tier_map) | NGL_CURRENT):
        if state not in STATE_CENTROIDS:
            continue
        lat, lon = STATE_CENTROIDS[state]
        ax.text(lon, lat, state, ha="center", va="center", fontsize=8, fontweight="bold", color="#111827", zorder=3)
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F8FAFC")
    ax.set_title("NGL Portfolio vs Final Reinforcement Candidates", fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_final_reinforcement_ngl_overlay_map.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    # Simple table image
    table_df = state_roles[["state_abbr", "tier", "role", "top3_commodities", "family_count", "scenario_count_total"]].copy()
    table_df = table_df.rename(
        columns={
            "state_abbr": "State",
            "tier": "Tier",
            "role": "Role",
            "top3_commodities": "Top commodities",
            "family_count": "Families",
            "scenario_count_total": "Scenarios",
        }
    )
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold")
        elif c == 1 and r > 0:
            tier = table_df.iloc[r - 1]["Tier"]
            cell.set_facecolor(TIER_COLOR.get(tier, "#FFFFFF"))
            cell.set_text_props(color="white" if tier != "Conditional watchlist" else "black", weight="bold")
    fig.suptitle("Final Reinforcement Candidate Table", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_final_reinforcement_candidate_table.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Interactive html overlay map
    fig = go.Figure()
    cat_order = [
        "Other state",
        "Other NGL state",
        "Conditional watchlist",
        "Secondary internal reinforcement",
        "Robust internal reinforcement",
        "Core expansion",
    ]
    df_map = pd.DataFrame({"state_abbr": ALL_STATES})
    categories = []
    colors = {
        "Other state": "#E5E7EB",
        "Other NGL state": "#CBD5E1",
        **TIER_COLOR,
    }
    for s in df_map["state_abbr"]:
        if s in tier_map:
            categories.append(tier_map[s])
        elif s in NGL_CURRENT:
            categories.append("Other NGL state")
        else:
            categories.append("Other state")
    df_map["category"] = categories
    for idx, cat in enumerate(cat_order):
        sub = df_map[df_map["category"] == cat]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, colors[cat]], [1, colors[cat]]],
                showscale=False,
                showlegend=True,
                name=cat,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + cat + "</extra>",
            )
        )
    labels = sorted((set(tier_map) | NGL_CURRENT) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labels],
            lon=[STATE_CENTROIDS[s][1] for s in labels],
            mode="text",
            text=labels,
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Final Reinforcement Candidates and NGL Portfolio",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / "04_final_reinforcement_candidate_map.html"), include_plotlyjs="cdn")

    metadata = {
        "tier_map_png": str(OUT_DIR / "01_final_reinforcement_tier_map.png"),
        "tier_map_html": str(OUT_DIR / "01_final_reinforcement_tier_map.html"),
        "overlay_map_png": str(OUT_DIR / "02_final_reinforcement_ngl_overlay_map.png"),
        "table_png": str(OUT_DIR / "03_final_reinforcement_candidate_table.png"),
        "html_map": str(OUT_DIR / "04_final_reinforcement_candidate_map.html"),
    }
    (OUT_DIR / "final_reinforcement_presentation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
TRADE_NEXT15 = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_enhanced_shock" / "simulation_results" / "trade_external_enhanced_next15_excluding_must_have.csv"
DOMESTIC_NEXT15 = WORK_ROOT / "outputs" / "scenario_model" / "domestic_broad_demand_shock" / "simulation_results" / "domestic_broad_demand_next15_excluding_must_have.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "enhanced_scenario_overlap_maps"

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


def category_for_state(state: str, scenario_states: set[str]) -> str:
    in_ngl = state in NGL_CURRENT
    in_scenario = state in scenario_states
    if in_ngl and in_scenario:
        return "Overlap"
    if in_scenario and not in_ngl:
        return "Scenario-only"
    if in_ngl and not in_scenario:
        return "NGL-only"
    return "Neither"


def build_sets() -> dict[str, set[str]]:
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))
    trade = must_have | set(pd.read_csv(TRADE_NEXT15)["state_abbr"].astype(str))
    domestic = must_have | set(pd.read_csv(DOMESTIC_NEXT15)["state_abbr"].astype(str))
    common = must_have | (trade - must_have) & (domestic - must_have)
    return {
        "must_have": must_have,
        "trade": trade,
        "domestic": domestic,
        "common": common,
    }


def make_html_map(name: str, title: str, scenario_states: set[str]) -> None:
    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df["category"] = df["state_abbr"].map(lambda s: category_for_state(s, scenario_states))
    color_map = {
        "Overlap": "#2563EB",
        "Scenario-only": "#16A34A",
        "NGL-only": "#F59E0B",
        "Neither": "#E5E7EB",
    }
    fig = go.Figure()
    for label, value in [("Neither", 0), ("NGL-only", 1), ("Scenario-only", 2), ("Overlap", 3)]:
        sub = df.loc[df["category"] == label]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[value] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, color_map[label]], [1, color_map[label]]],
                showscale=False,
                showlegend=True,
                name=label,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + label + "</extra>",
            )
        )
    labeled = sorted((scenario_states | NGL_CURRENT) & set(STATE_CENTROIDS))
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
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / f"{name}.html"), include_plotlyjs="cdn")


def make_static_maps(sets: dict[str, set[str]]) -> None:
    color_map = {
        "Overlap": "#2563EB",
        "Scenario-only": "#16A34A",
        "NGL-only": "#F59E0B",
        "Neither": "#E5E7EB",
    }
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label="Overlap", markerfacecolor=color_map["Overlap"], markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Scenario-only", markerfacecolor=color_map["Scenario-only"], markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="NGL-only", markerfacecolor=color_map["NGL-only"], markeredgecolor="white", markersize=10),
        plt.Line2D([0], [0], marker="o", color="w", label="Neither", markerfacecolor=color_map["Neither"], markeredgecolor="white", markersize=10),
    ]
    panels = [
        ("01_enhanced_trade_ngl_overlap", "Enhanced Trade vs NGL", sets["trade"]),
        ("02_domestic_broad_demand_ngl_overlap", "Domestic Broad Demand vs NGL", sets["domestic"]),
        ("03_common_enhanced_candidates_ngl_overlap", "Common Enhanced Candidates vs NGL", sets["common"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    for ax, (_, title, scenario_states) in zip(axes, panels):
        for state in ALL_CONTIG:
            lat, lon = STATE_CENTROIDS[state]
            cat = category_for_state(state, scenario_states)
            ax.scatter(lon, lat, s=210, color=color_map[cat], edgecolor="white", linewidth=0.8, zorder=2)
        labels = sorted((scenario_states | NGL_CURRENT) & set(STATE_CENTROIDS))
        for state in labels:
            lat, lon = STATE_CENTROIDS[state]
            ax.text(lon, lat, state, ha="center", va="center", fontsize=7, fontweight="bold", color="#111827", zorder=3)
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#F8FAFC")
        ax.set_title(title, fontsize=12, weight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=10)
    fig.suptitle("NGL Portfolio vs Enhanced Scenario Results", fontsize=16, weight="bold", y=0.97)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT_DIR / "04_enhanced_scenario_ngl_static_comparison.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    for name, title, scenario_states in panels:
        fig_single, ax = plt.subplots(figsize=(10, 7))
        for state in ALL_CONTIG:
            lat, lon = STATE_CENTROIDS[state]
            cat = category_for_state(state, scenario_states)
            ax.scatter(lon, lat, s=230, color=color_map[cat], edgecolor="white", linewidth=0.8, zorder=2)
        labels = sorted((scenario_states | NGL_CURRENT) & set(STATE_CENTROIDS))
        for state in labels:
            lat, lon = STATE_CENTROIDS[state]
            ax.text(lon, lat, state, ha="center", va="center", fontsize=8, fontweight="bold", color="#111827", zorder=3)
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("#F8FAFC")
        ax.set_title(title, fontsize=14, weight="bold")
        ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9)
        fig_single.tight_layout()
        fig_single.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
        plt.close(fig_single)


def write_summary(sets: dict[str, set[str]]) -> None:
    lines = ["# Enhanced Scenario vs NGL Overlap Summary", ""]
    for key, title in [
        ("trade", "Enhanced Trade"),
        ("domestic", "Domestic Broad Demand"),
        ("common", "Common Enhanced Candidates"),
    ]:
        scenario_states = sets[key]
        overlap = sorted(NGL_CURRENT & scenario_states)
        scenario_only = sorted(scenario_states - NGL_CURRENT)
        ngl_only = sorted(NGL_CURRENT - scenario_states)
        lines.extend([
            f"## {title}",
            "",
            f"- overlap count: {len(overlap)}",
            f"- scenario-only count: {len(scenario_only)}",
            f"- ngl-only count: {len(ngl_only)}",
            "",
            "### Overlap",
            "- " + ", ".join(overlap),
            "",
            "### Scenario-only",
            "- " + ", ".join(scenario_only) if scenario_only else "- none",
            "",
            "### NGL-only",
            "- " + ", ".join(ngl_only),
            "",
        ])
    (OUT_DIR / "enhanced_scenario_ngl_overlap_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = build_sets()
    make_html_map("01_enhanced_trade_ngl_overlap", "NGL vs Enhanced Trade Shock", sets["trade"])
    make_html_map("02_domestic_broad_demand_ngl_overlap", "NGL vs Domestic Broad Demand Shock", sets["domestic"])
    make_html_map("03_common_enhanced_candidates_ngl_overlap", "NGL vs Common Enhanced Candidates", sets["common"])
    make_static_maps(sets)
    write_summary(sets)
    metadata = {
        "trade_html": str(OUT_DIR / "01_enhanced_trade_ngl_overlap.html"),
        "domestic_html": str(OUT_DIR / "02_domestic_broad_demand_ngl_overlap.html"),
        "common_html": str(OUT_DIR / "03_common_enhanced_candidates_ngl_overlap.html"),
        "static_panel_png": str(OUT_DIR / "04_enhanced_scenario_ngl_static_comparison.png"),
        "summary_md": str(OUT_DIR / "enhanced_scenario_ngl_overlap_summary.md"),
    }
    (OUT_DIR / "enhanced_scenario_ngl_overlap_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

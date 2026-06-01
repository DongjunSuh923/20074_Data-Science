from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
BASELINE_NEXT10_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_hubs" / "scenario_next10_excluding_must_have.csv"
TIER_A_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "tier_a_hub_candidates" / "tier_a_final_candidates.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_strategy_comparison_maps"

ALL_CONTIG = [
    "AL","AZ","AR","CA","CO","CT","DE","DC","FL","GA","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",
    "TN","TX","UT","VT","VA","WA","WV","WI","WY"
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

NGL_CURRENT = {
    "AL","AZ","CA","CO","CT","FL","IA","IL","IN","KS","KY","LA","MA","MI","MN","MO","MS","NE","NV",
    "NY","OH","OK","OR","PA","RI","SD","TN","TX","WA","WI"
}


def draw_panel(ax, title: str, category_map: dict[str, str], color_map: dict[str, str]) -> None:
    for state in ALL_CONTIG:
        lat, lon = STATE_CENTROIDS[state]
        cat = category_map.get(state, "Other")
        ax.scatter(lon, lat, s=60, color=color_map.get(cat, "#D1D5DB"), edgecolor="white", linewidth=0.6, zorder=2)
    for state, cat in category_map.items():
        if state not in STATE_CENTROIDS:
            continue
        lat, lon = STATE_CENTROIDS[state]
        ax.text(lon, lat + 0.7, state, ha="center", va="center", fontsize=8, fontweight="bold", color="#111827", zorder=3)
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlim(-125, -66)
    ax.set_ylim(24, 50)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#F8FAFC")


def add_legend(ax, items: list[tuple[str, str]]) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markeredgecolor="white", markersize=8)
        for label, color in items
    ]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))
    baseline_next10 = set(
        pd.read_csv(BASELINE_NEXT10_PATH)
        .loc[lambda d: d["scenario_name"] == "baseline_no_shock", "state_abbr"]
        .astype(str)
    )
    tier_a = set(pd.read_csv(TIER_A_PATH)["state_abbr"].astype(str))

    ngl_map = {s: "NGL Terminal" for s in NGL_CURRENT if s in STATE_CENTROIDS}
    baseline_map = {s: "Must-have Core" for s in must_have if s in STATE_CENTROIDS}
    baseline_map.update({s: "Baseline Next 10" for s in baseline_next10 if s in STATE_CENTROIDS})
    scenario_map = {s: "Must-have Core" for s in must_have if s in STATE_CENTROIDS}
    scenario_map.update({s: "Tier A Expansion" for s in tier_a if s in STATE_CENTROIDS})

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    draw_panel(axes[0], "NGL Current Terminal Footprint", ngl_map, {"NGL Terminal": "#2563EB"})
    add_legend(axes[0], [("NGL Terminal", "#2563EB"), ("Other", "#D1D5DB")])

    draw_panel(axes[1], "Pre-Scenario Model Importance", baseline_map, {"Must-have Core": "#B91C1C", "Baseline Next 10": "#F59E0B"})
    add_legend(axes[1], [("Must-have Core", "#B91C1C"), ("Baseline Next 10", "#F59E0B"), ("Other", "#D1D5DB")])

    draw_panel(axes[2], "Scenario Final Placement", scenario_map, {"Must-have Core": "#B91C1C", "Tier A Expansion": "#16A34A"})
    add_legend(axes[2], [("Must-have Core", "#B91C1C"), ("Tier A Expansion", "#16A34A"), ("Other", "#D1D5DB")])

    fig.suptitle("State-Level Hub Strategy Comparison", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / "04_hub_strategy_static_comparison.png"
    fig.savefig(out, bbox_inches="tight", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()

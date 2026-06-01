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
from matplotlib.patches import FancyBboxPatch


IN_CSV = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "ngl_commodity_recommendations" / "ngl_commodity_specific_recommendations.csv"
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep"

LABEL_MAP = {
    "Cereal grains and agricultural support": "Cereal grains",
    "Construction materials and aggregates": "Construction materials / aggregates",
    "Logs and wood-linked flows": "Logs / wood-linked",
    "Trade redistribution and gateway support": "Trade redistribution / gateway",
    "Fuel and energy-related balancing": "Fuel / energy balancing",
    "Shock-specific trade gateway reinforcement": "Trade-specific adjunct",
}

PRIORITY_COLORS = {
    "Primary": "#2563EB",
    "Secondary": "#F59E0B",
    "Adjunct": "#0F766E",
}


def wrap_states(text: str) -> str:
    return text.replace(", ", ",\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df["label"] = df["commodity_theme"].map(LABEL_MAP).fillna(df["commodity_theme"])

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        0.03, 0.965,
        "Commodity-Specific NGL Reinforcement Proposal",
        fontsize=20, fontweight="bold", color="#111827", va="top",
    )
    ax.text(
        0.03, 0.93,
        "Not just where to add hubs, but which states reinforce which commodity and shock mechanism.",
        fontsize=11.5, color="#4B5563", va="top",
    )

    headers = [("Commodity theme", 0.03), ("Priority", 0.35), ("Recommended states", 0.47), ("Supporting states", 0.76)]
    for title, x in headers:
        ax.text(x, 0.875, title, fontsize=12, fontweight="bold", color="#374151", va="center")

    top_y = 0.82
    row_h = 0.115

    for i, row in enumerate(df.itertuples(index=False)):
        y = top_y - i * row_h
        bg = FancyBboxPatch(
            (0.025, y - 0.08), 0.95, 0.09,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.8, edgecolor="#E5E7EB", facecolor="#F8FAFC"
        )
        ax.add_patch(bg)

        ax.text(0.03, y - 0.015, row.label, fontsize=12.5, color="#111827", va="center")

        badge = FancyBboxPatch(
            (0.35, y - 0.045), 0.09, 0.04,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=0, facecolor=PRIORITY_COLORS[row.priority_group]
        )
        ax.add_patch(badge)
        ax.text(0.395, y - 0.025, row.priority_group, fontsize=10.5, color="white", fontweight="bold", ha="center", va="center")

        ax.text(0.47, y - 0.015, wrap_states(row.recommended_states), fontsize=12, color="#111827", va="center")
        support = row.supporting_states if isinstance(row.supporting_states, str) and row.supporting_states.strip() else "-"
        ax.text(0.76, y - 0.015, wrap_states(support), fontsize=12, color="#111827", va="center")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "15_commodity_specific_recommendation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

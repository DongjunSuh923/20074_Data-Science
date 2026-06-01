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


IN_CSV = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "final_reinforcement_candidates" / "final_reinforcement_candidate_set.csv"
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep"

TIER_ORDER = [
    "Core expansion",
    "Robust internal reinforcement",
    "Secondary internal reinforcement",
]

TIER_COLORS = {
    "Core expansion": "#F59E0B",
    "Robust internal reinforcement": "#2563EB",
    "Secondary internal reinforcement": "#10B981",
}

TITLE_MAP = {
    "Core expansion": "Core expansion",
    "Robust internal reinforcement": "Robust internal reinforcement",
    "Secondary internal reinforcement": "Secondary internal reinforcement",
}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN_CSV)
    df = df[df["tier"].isin(TIER_ORDER)].copy()

    grouped = {
        tier: ", ".join(df.loc[df["tier"] == tier, "state_abbr"].tolist())
        for tier in TIER_ORDER
    }

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    boxes = [
        (0.06, 0.68, 0.88, 0.22, "Core expansion"),
        (0.06, 0.39, 0.88, 0.22, "Robust internal reinforcement"),
        (0.06, 0.10, 0.88, 0.22, "Secondary internal reinforcement"),
    ]

    for x, y, w, h, tier in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            linewidth=1.5,
            edgecolor=TIER_COLORS[tier],
            facecolor="#F8FAFC",
        )
        ax.add_patch(patch)
        ax.text(x + 0.03, y + h - 0.07, TITLE_MAP[tier], fontsize=15, fontweight="bold", color=TIER_COLORS[tier])
        ax.text(x + 0.03, y + 0.07, grouped[tier], fontsize=18, color="#111827")

    ax.text(
        0.06,
        0.96,
        "Final Reinforcement Candidate Tiers",
        fontsize=19,
        fontweight="bold",
        color="#111827",
        va="top",
    )
    ax.text(
        0.06,
        0.925,
        "New expansion first: NC, VA | Internal reinforcement first: WA, MN, NE, KS, CO",
        fontsize=11.5,
        color="#4B5563",
        va="top",
    )

    plt.tight_layout()
    fig.savefig(OUT_DIR / "14_final_reinforcement_tier_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

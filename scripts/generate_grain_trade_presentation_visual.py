from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#D1D5DB", title_color="#111827"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.02, y + height - 0.055, title, fontsize=13, weight="bold", color=title_color, va="top")
    ax.text(x + 0.02, y + height - 0.11, body, fontsize=10.5, color="#1F2937", va="top", linespacing=1.45)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.5, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle(
        "Grain Supply vs Trade / External Shock: What Each Scenario Is Really Showing",
        fontsize=17,
        weight="bold",
        y=0.98,
    )

    # Left column: grain supply
    add_box(
        ax,
        (0.04, 0.56),
        0.42,
        0.34,
        "1. Grain supply shock",
        "Selected states:\n"
        "- IA severe\n- TX, NE medium\n- IL, MN mild\n\n"
        "Interpretation:\n"
        "- A commodity-specific supply shock\n"
        "- Not a broad national network breakdown\n"
        "- Strong mainly inside cereal flows",
        facecolor="#EFF6FF",
        edgecolor="#93C5FD",
    )
    add_box(
        ax,
        (0.04, 0.18),
        0.42,
        0.30,
        "2. What changed most",
        "Cumulative network loss: 0.19%\n"
        "Cereal-only cumulative loss: 2.42%\n\n"
        "Key reinforcement reading:\n"
        "- MN, NE, KS: grain support core\n"
        "- WA, OR: Pacific Northwest export support",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
    )

    # Right column: trade / external
    add_box(
        ax,
        (0.54, 0.56),
        0.42,
        0.34,
        "3. Enhanced trade / external shock",
        "Selected states:\n"
        "- CA severe\n- GA, NJ medium\n- MI, IL mild\n\n"
        "Why enhanced:\n"
        "- Added imports\n"
        "- Added container exposure\n"
        "- Added partner-country trade-war exposure",
        facecolor="#FFF7ED",
        edgecolor="#FDBA74",
    )
    add_box(
        ax,
        (0.54, 0.18),
        0.42,
        0.30,
        "4. What this means",
        "Cumulative network loss: 0.32%\n"
        "Interstate cumulative loss: 1.54%\n\n"
        "Key reinforcement reading:\n"
        "- NJ, GA: gateway reinforcement\n"
        "- NC, VA: East Coast inland connector reinforcement",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
    )

    ax.text(
        0.5,
        0.08,
        "Takeaway: Grain shock is a narrow, commodity-specific disruption, while enhanced trade shock captures\n"
        "import + container + partner-country exposure and therefore highlights gateway and inland redistribution roles.",
        ha="center",
        va="center",
        fontsize=11.2,
        color="#374151",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "09_grain_trade_scenario_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

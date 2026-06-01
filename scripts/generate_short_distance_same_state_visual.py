from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


READ_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

METRICS_PATH = READ_ROOT / "outputs" / "scenario_model" / "short_distance_analysis" / "short_distance_same_state_metrics.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(METRICS_PATH)

    def get_value(split: str, group: str, col: str) -> float:
        return float(df.loc[(df["split"] == split) & (df["same_state_group"] == group), col].iloc[0])

    splits = ["test_2023", "test_2024"]
    split_labels = ["2023", "2024"]

    actual_same = [get_value(s, "same_state", "actual_share_pct") for s in splits]
    actual_cross = [get_value(s, "cross_state", "actual_share_pct") for s in splits]
    error_same = [get_value(s, "same_state", "abs_error_share_pct") for s in splits]
    error_cross = [get_value(s, "cross_state", "abs_error_share_pct") for s in splits]

    x = np.arange(len(split_labels))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Actual share panel
    ax = axes[0]
    bars1 = ax.bar(x - width / 2, actual_same, width, color="#4E79A7", label="Same-state")
    bars2 = ax.bar(x + width / 2, actual_cross, width, color="#F28E2B", label="Cross-state")
    ax.set_title("Short-Distance Actual Share", fontsize=14, weight="bold")
    ax.set_ylabel("Share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    # Error share panel
    ax = axes[1]
    bars3 = ax.bar(x - width / 2, error_same, width, color="#4E79A7", label="Same-state")
    bars4 = ax.bar(x + width / 2, error_cross, width, color="#F28E2B", label="Cross-state")
    ax.set_title("Short-Distance Error Share", fontsize=14, weight="bold")
    ax.set_ylabel("Share (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(split_labels)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    for bars in (bars3, bars4):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Short-Distance Flows Are Dominated by Same-State Movement", fontsize=16, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "02_short_distance_same_state_vs_cross_state.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

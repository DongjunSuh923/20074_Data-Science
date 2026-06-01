from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


READ_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

DIST_BAND_PATH = READ_ROOT / "outputs" / "scenario_model" / "diagnostics_fhwa_gravel_best" / "fhwa_gravel_best_compare_2024_vs_2023_by_dist_band.csv"
SAME_STATE_PATH = READ_ROOT / "outputs" / "scenario_model" / "short_distance_analysis" / "short_distance_same_state_metrics.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dist = pd.read_csv(DIST_BAND_PATH)
    same = pd.read_csv(SAME_STATE_PATH)

    dist["dist_band"] = dist["dist_band"].astype(str)
    dist = dist.sort_values("dist_band", key=lambda s: s.astype(int))

    x = np.arange(len(dist))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))

    # Left: dist band RMSE
    ax = axes[0]
    bars_2023 = ax.bar(x - width / 2, dist["rmse_2023"], width, color="#4E79A7", label="2023")
    bars_2024 = ax.bar(x + width / 2, dist["rmse_2024"], width, color="#F28E2B", label="2024")
    ax.set_title("RMSE by Distance Band", fontsize=14, weight="bold")
    ax.set_xlabel("Distance band")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}" for d in dist["dist_band"]])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    # highlight short-distance area
    ax.axvspan(-0.5, 1.5, color="#E5E7EB", alpha=0.35, zorder=0)
    ax.text(
        0.5,
        max(dist["rmse_2024"]) * 1.03,
        "Short-distance zone\n(dist_band 1 & 2)",
        ha="center",
        va="bottom",
        fontsize=10,
        color="#374151",
    )

    # Right: short-distance actual share
    ax = axes[1]
    splits = ["2023", "2024"]
    x2 = np.arange(len(splits))
    actual_same = [
        float(same.loc[(same["split"] == "test_2023") & (same["same_state_group"] == "same_state"), "actual_share_pct"].iloc[0]),
        float(same.loc[(same["split"] == "test_2024") & (same["same_state_group"] == "same_state"), "actual_share_pct"].iloc[0]),
    ]
    actual_cross = [
        float(same.loc[(same["split"] == "test_2023") & (same["same_state_group"] == "cross_state"), "actual_share_pct"].iloc[0]),
        float(same.loc[(same["split"] == "test_2024") & (same["same_state_group"] == "cross_state"), "actual_share_pct"].iloc[0]),
    ]
    bars_same = ax.bar(x2 - width / 2, actual_same, width, color="#4E79A7", label="Same-state")
    bars_cross = ax.bar(x2 + width / 2, actual_cross, width, color="#F28E2B", label="Cross-state")
    ax.set_title("Short-Distance Actual Share", fontsize=14, weight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)")
    ax.set_xticks(x2)
    ax.set_xticklabels(splits)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (bars_same, bars_cross):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.2, f"{h:.1f}%", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Short-Distance Freight Is Concentrated in the Shortest Bands and Dominated by Same-State Flows", fontsize=15, weight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / "03_dist_band_and_short_distance_actual_share.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    splits = ["Validation", "Test 2023", "Test 2024"]
    baseline = [290.19, 313.26, 383.66]
    current_best = [252.98, 273.95, 326.04]

    x = np.arange(len(splits))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2,
        baseline,
        width,
        color="#F28E2B",
        label="Midterm-Time Baseline",
    )
    bars2 = ax.bar(
        x + width / 2,
        current_best,
        width,
        color="#4E79A7",
        label="Current Best (Pre-simulation)",
    )

    ax.set_title("Model Performance: Midterm-Time Baseline vs Current Best", fontsize=15, weight="bold")
    ax.set_ylabel("RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 4,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_model_performance_baseline_vs_current_best.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


READ_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

SUMMARY_PATH = (
    READ_ROOT
    / "outputs"
    / "scenario_model"
    / "natural_disaster_capacity"
    / "simulation_results"
    / "natural_disaster_simulation_summary.csv"
)
NEXT15_DIR = READ_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"

SCENARIO_LABELS = {
    "snow_heat_state_blocking": "Snow / Heat",
    "tornado_state_blocking": "Tornado",
    "wildfire_smoke_state_blocking": "Wildfire / Smoke",
}

SCENARIO_FILES = {
    "snow_heat_state_blocking": NEXT15_DIR / "snow_heat_state_blocking_next15_excluding_must_have.csv",
    "tornado_state_blocking": NEXT15_DIR / "tornado_state_blocking_next15_excluding_must_have.csv",
    "wildfire_smoke_state_blocking": NEXT15_DIR / "wildfire_smoke_state_blocking_next15_excluding_must_have.csv",
}


def build_repeat_table() -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for scenario_name, path in SCENARIO_FILES.items():
        frame = pd.read_csv(path)
        frame["scenario_name"] = scenario_name
        rows.extend(frame[["state_abbr", "next_rank", "scenario_name"]].to_dict("records"))

    combined = pd.DataFrame(rows)
    repeat = (
        combined.groupby("state_abbr", as_index=False)
        .agg(
            repeat_count=("scenario_name", "nunique"),
            avg_rank=("next_rank", "mean"),
        )
        .sort_values(["repeat_count", "avg_rank", "state_abbr"], ascending=[False, True, True])
    )
    return repeat


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    summary = summary.copy()
    summary["label"] = summary["scenario_name"].map(SCENARIO_LABELS)
    summary["loss_pct"] = summary["cumulative_network_loss_pct_of_annual_sum"] * 100

    repeat = build_repeat_table()
    repeat = repeat[repeat["repeat_count"] >= 2].head(8).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cumulative loss by natural disaster scenario
    ax = axes[0]
    colors = ["#4E79A7", "#F28E2B", "#59A14F"]
    bars = ax.bar(summary["label"], summary["loss_pct"], color=colors)
    ax.set_title("Natural Disaster Shock by Scenario", fontsize=14, weight="bold")
    ax.set_ylabel("Cumulative network loss (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(summary["loss_pct"]) * 1.28)

    for bar, blocked in zip(bars, summary["blocked_states"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.55,
            f"Blocked:\n{blocked}",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1F2937",
        )

    # Right: repeated backup candidates across natural-disaster scenarios
    ax = axes[1]
    repeat = repeat.sort_values(["repeat_count", "avg_rank"], ascending=[True, False])
    bar_colors = ["#4E79A7"] * len(repeat)
    if "AR" in repeat["state_abbr"].values:
        ar_idx = repeat.index[repeat["state_abbr"] == "AR"][0]
        bar_colors[list(repeat.index).index(ar_idx)] = "#F28E2B"

    bars = ax.barh(repeat["state_abbr"], repeat["repeat_count"], color=bar_colors)
    ax.set_title("Repeated Backup Candidates\n(Appear in 2+ Natural-Disaster Scenarios)", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario count")
    ax.set_xlim(0, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.grid(axis="x", alpha=0.25)

    for bar, avg_rank, state in zip(bars, repeat["avg_rank"], repeat["state_abbr"]):
        width = bar.get_width()
        note = f"{int(width)} scenarios"
        if state == "AR":
            note += f"  | avg rank {avg_rank:.1f}"
        ax.text(width + 0.07, bar.get_y() + bar.get_height() / 2, note, va="center", fontsize=9)

    ax.text(
        0.02,
        -0.18,
        "AR is highlighted because it is not a baseline core hub,\n"
        "but it reappears as a south-central fallback candidate.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
    )

    fig.suptitle(
        "Natural-Disaster Scenarios Create the Broadest Network Stress and Reveal Fallback States",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / "04_natural_disaster_scenario_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

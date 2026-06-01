from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DOWNLOAD_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep"

CONSTRUCTION_SUMMARY = IDEA_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid" / "construction_infra_hybrid_summary.csv"
BOTTLENECK_SUMMARY = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_summary.csv"
BROAD_DEMAND_SUMMARY = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "domestic_broad_demand_shock" / "simulation_results" / "domestic_broad_demand_summary.csv"

CONSTRUCTION_NEXT15 = IDEA_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid" / "construction_infra_hybrid_next15_excluding_must_have.csv"
BOTTLENECK_NEXT15 = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_next15_excluding_must_have.csv"
BROAD_DEMAND_NEXT15 = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "domestic_broad_demand_shock" / "simulation_results" / "domestic_broad_demand_next15_excluding_must_have.csv"


def build_repeat_table() -> pd.DataFrame:
    frames = []
    for scenario_name, path in [
        ("Construction / Infrastructure", CONSTRUCTION_NEXT15),
        ("Bottleneck / Congestion", BOTTLENECK_NEXT15),
        ("Broad Domestic Demand", BROAD_DEMAND_NEXT15),
    ]:
        frame = pd.read_csv(path)
        frame = frame.loc[frame["next_rank"] <= 15, ["state_abbr", "next_rank"]].copy()
        frame["scenario_name"] = scenario_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
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

    construction = pd.read_csv(CONSTRUCTION_SUMMARY).iloc[0]
    bottleneck = pd.read_csv(BOTTLENECK_SUMMARY).iloc[0]
    broad = pd.read_csv(BROAD_DEMAND_SUMMARY).iloc[0]

    loss_df = pd.DataFrame(
        [
            {
                "label": "Construction /\nInfrastructure",
                "cumulative_network_loss_pct": float(construction["cumulative_network_loss_pct_of_annual_sum"]) * 100,
                "secondary_metric_pct": float(construction["cumulative_subset_loss_pct_of_annual_sum"]) * 100,
                "secondary_label": "Material-subset loss",
            },
            {
                "label": "Bottleneck /\nCongestion",
                "cumulative_network_loss_pct": float(bottleneck["cumulative_network_loss_pct_of_annual_sum"]) * 100,
                "secondary_metric_pct": float(bottleneck["cumulative_interstate_loss_pct_of_annual_sum"]) * 100,
                "secondary_label": "Interstate loss",
            },
            {
                "label": "Broad Domestic\nDemand",
                "cumulative_network_loss_pct": float(broad["cumulative_network_loss_pct_of_annual_sum"]) * 100,
                "secondary_metric_pct": float(broad["cumulative_interstate_loss_pct_of_annual_sum"]) * 100,
                "secondary_label": "Interstate loss",
            },
        ]
    )

    repeat = build_repeat_table()
    repeat = repeat[repeat["repeat_count"] >= 2].head(10).copy()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))

    # Left: domestic issue scenario comparison
    ax = axes[0]
    bars = ax.bar(
        loss_df["label"],
        loss_df["cumulative_network_loss_pct"],
        color=["#4E79A7", "#F28E2B", "#59A14F"],
    )
    ax.set_title("Domestic-Issue Shock by Scenario", fontsize=14, weight="bold")
    ax.set_ylabel("Cumulative network loss (%)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(loss_df["cumulative_network_loss_pct"]) * 1.38)

    for bar, row in zip(bars, loss_df.itertuples(index=False)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.08,
            f"{height:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height * 0.57,
            f"{row.secondary_label}\n{row.secondary_metric_pct:.2f}%",
            ha="center",
            va="center",
            fontsize=8.8,
            color="#1F2937",
        )

    # Right: repeated reinforcement candidates
    ax = axes[1]
    repeat = repeat.sort_values(["repeat_count", "avg_rank"], ascending=[True, False])
    bar_colors = []
    for state in repeat["state_abbr"]:
        if state in {"NC", "VA"}:
            bar_colors.append("#F28E2B")
        else:
            bar_colors.append("#4E79A7")

    bars = ax.barh(repeat["state_abbr"], repeat["repeat_count"], color=bar_colors)
    ax.set_title("Repeated Reinforcement Candidates\n(Appear in 2+ Domestic-Issue Scenarios)", fontsize=14, weight="bold")
    ax.set_xlabel("Scenario count")
    ax.set_xlim(0, 3.4)
    ax.set_xticks([1, 2, 3])
    ax.grid(axis="x", alpha=0.25)

    for bar, avg_rank, state in zip(bars, repeat["avg_rank"], repeat["state_abbr"]):
        width = bar.get_width()
        label = f"{int(width)} scenarios"
        if state in {"NC", "VA"}:
            label += f"  | avg rank {avg_rank:.1f}"
        ax.text(width + 0.07, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)

    ax.text(
        0.02,
        -0.18,
        "NC and VA are highlighted because they survive across\n"
        "construction, bottleneck, and broader-demand scenarios.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#374151",
    )

    fig.suptitle(
        "Domestic-Issue Scenarios Split into Demand Weakening, Connectivity Stress, and Broad Demand Weakening",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(OUT_DIR / "07_domestic_issue_scenario_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

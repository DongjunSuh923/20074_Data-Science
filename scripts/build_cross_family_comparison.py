from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "cross_family_comparison"
MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"

NGL_CURRENT = {
    "AL","AZ","CA","CO","CT","FL","IA","IL","IN","KS","KY","LA","MA","MI","MN","MO","MS","NE","NV",
    "NY","OH","OK","OR","PA","RI","SD","TN","TX","WA","WI"
}


def load_scenario_candidates() -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []

    # Natural disaster + grain from the shared next15 table built earlier.
    shared = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs" / "scenario_next15_excluding_must_have.csv")
    keep = {
        "snow_heat_state_blocking": ("Capacity", "snow_heat"),
        "tornado_state_blocking": ("Capacity", "tornado"),
        "wildfire_smoke_state_blocking": ("Capacity", "wildfire_smoke"),
        "grain_supply_state_blocking": ("Commodity-specific supply", "grain_supply"),
    }
    for _, rec in shared.iterrows():
        scenario_name = str(rec["scenario_name"])
        if scenario_name not in keep:
            continue
        family, short = keep[scenario_name]
        rows.append(
            {
                "family": family,
                "scenario": short,
                "state_abbr": str(rec["state_abbr"]),
                "next_rank": int(rec["next_rank"]),
            }
        )

    # Domestic construction hybrid
    construction = pd.read_csv(
        PROJECT_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid" / "construction_infra_hybrid_next15_excluding_must_have.csv"
    )
    for _, rec in construction.iterrows():
        rows.append(
            {
                "family": "Demand",
                "scenario": "construction_hybrid",
                "state_abbr": str(rec["state_abbr"]),
                "next_rank": int(rec["next_rank"]),
            }
        )

    # Domestic bottleneck
    bottleneck = pd.read_csv(
        WORK_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock" / "domestic_bottleneck_next15_excluding_must_have.csv"
    )
    for _, rec in bottleneck.iterrows():
        rows.append(
            {
                "family": "Demand",
                "scenario": "domestic_bottleneck",
                "state_abbr": str(rec["state_abbr"]),
                "next_rank": int(rec["next_rank"]),
            }
        )

    # Domestic broad demand
    broad = pd.read_csv(
        WORK_ROOT / "outputs" / "scenario_model" / "domestic_broad_demand_shock" / "simulation_results" / "domestic_broad_demand_next15_excluding_must_have.csv"
    )
    for _, rec in broad.iterrows():
        rows.append(
            {
                "family": "Demand",
                "scenario": "domestic_broad_demand",
                "state_abbr": str(rec["state_abbr"]),
                "next_rank": int(rec["next_rank"]),
            }
        )

    # Enhanced trade only
    trade = pd.read_csv(
        WORK_ROOT / "outputs" / "scenario_model" / "trade_external_enhanced_shock" / "simulation_results" / "trade_external_enhanced_next15_excluding_must_have.csv"
    )
    for _, rec in trade.iterrows():
        rows.append(
            {
                "family": "Trade/external network",
                "scenario": "trade_external_enhanced",
                "state_abbr": str(rec["state_abbr"]),
                "next_rank": int(rec["next_rank"]),
            }
        )

    return pd.DataFrame(rows)


def build_outputs(df: pd.DataFrame) -> dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))

    matrix = (
        df.assign(present=1)
        .pivot_table(index="state_abbr", columns="scenario", values="present", aggfunc="max", fill_value=0)
        .reset_index()
    )
    matrix["scenario_count_matrix"] = matrix.drop(columns=["state_abbr"]).sum(axis=1)
    family_presence = (
        df.assign(present=1)
        .groupby(["state_abbr", "family"], as_index=False)["present"]
        .max()
        .pivot_table(index="state_abbr", columns="family", values="present", aggfunc="max", fill_value=0)
        .reset_index()
    )
    family_cols = [c for c in family_presence.columns if c != "state_abbr"]
    family_presence["family_count"] = family_presence[family_cols].sum(axis=1)
    rank_stats = df.groupby("state_abbr", as_index=False).agg(
        scenario_count_total=("scenario", "nunique"),
        avg_rank=("next_rank", "mean"),
        best_rank=("next_rank", "min"),
    )
    rank_stats["is_ngl_current"] = rank_stats["state_abbr"].isin(NGL_CURRENT)
    rank_stats["is_must_have"] = rank_stats["state_abbr"].isin(must_have)
    combined = (
        rank_stats.merge(family_presence, on="state_abbr", how="left")
        .merge(matrix, on="state_abbr", how="left")
        .sort_values(["family_count", "scenario_count_total", "best_rank", "avg_rank"], ascending=[False, False, True, True])
    )
    combined.to_csv(OUT_DIR / "cross_family_candidate_matrix.csv", index=False)

    family_summary = (
        df.groupby(["family", "state_abbr"], as_index=False)
        .agg(scenario_count=("scenario", "nunique"), avg_rank=("next_rank", "mean"), best_rank=("next_rank", "min"))
        .sort_values(["family", "scenario_count", "best_rank", "avg_rank"], ascending=[True, False, True, True])
    )
    family_summary.to_csv(OUT_DIR / "cross_family_family_summary.csv", index=False)

    common = combined.loc[combined["family_count"] >= 2].copy()
    common.to_csv(OUT_DIR / "cross_family_common_candidates.csv", index=False)

    scenario_only = common.loc[(~common["is_ngl_current"]) & (~common["is_must_have"])].copy()
    scenario_only.to_csv(OUT_DIR / "cross_family_common_scenario_only_candidates.csv", index=False)

    ngl_only = sorted((NGL_CURRENT - must_have) - set(df["state_abbr"]))
    pd.DataFrame({"state_abbr": ngl_only}).to_csv(OUT_DIR / "cross_family_never_selected_ngl_only.csv", index=False)

    # Plot: top repeated candidates
    top = combined.head(20).copy()
    labels = top["state_abbr"]
    values = top["scenario_count_total"]
    colors = ["#2563EB" if s in NGL_CURRENT else "#16A34A" for s in labels]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(labels, values, color=colors)
    ax.set_title("Cross-Family Repeated Candidate Count", fontsize=15, weight="bold")
    ax.set_ylabel("Scenario Count")
    ax.set_xlabel("State")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_cross_family_candidate_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Plot: family heatmap-ish table via imshow
    top_states = common.head(18)["state_abbr"].tolist()
    heat = (
        df.loc[df["state_abbr"].isin(top_states)]
        .assign(present=1)
        .pivot_table(index="state_abbr", columns="scenario", values="present", aggfunc="max", fill_value=0)
        .reindex(top_states)
    )
    fig, ax = plt.subplots(figsize=(14, max(5, len(top_states) * 0.35)))
    ax.imshow(heat.values, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index)
    ax.set_title("Cross-Family Candidate Presence by Scenario", fontsize=15, weight="bold")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j, i, int(heat.values[i, j]), ha="center", va="center", fontsize=8, color="#111827")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_cross_family_presence_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_lines = [
        "# Cross-Family Scenario Comparison Summary",
        "",
        "Included scenarios:",
        "- Capacity: snow_heat, tornado, wildfire_smoke",
        "- Commodity-specific supply: grain_supply",
        "- Demand: construction_hybrid, domestic_bottleneck, domestic_broad_demand",
        "- Trade/external network: trade_external_enhanced",
        "",
        "## Most repeated candidates",
    ]
    for _, rec in combined.head(15).iterrows():
        summary_lines.append(
            f"- {rec['state_abbr']}: families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}, best_rank={int(rec['best_rank'])}, avg_rank={rec['avg_rank']:.2f}"
        )
    summary_lines.extend(
        [
            "",
            "## Common scenario-only candidates (non-NGL, non-must-have, family_count >= 2)",
        ]
    )
    if scenario_only.empty:
        summary_lines.append("- none")
    else:
        for _, rec in scenario_only.iterrows():
            summary_lines.append(
                f"- {rec['state_abbr']}: families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}, best_rank={int(rec['best_rank'])}, avg_rank={rec['avg_rank']:.2f}"
            )
    summary_lines.extend(
        [
            "",
            "## Non-must-have NGL-only states never selected in any included scenario",
            "- " + ", ".join(ngl_only) if ngl_only else "- none",
        ]
    )
    (OUT_DIR / "cross_family_comparison_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    metadata = {
        "matrix_csv": str(OUT_DIR / "cross_family_candidate_matrix.csv"),
        "family_summary_csv": str(OUT_DIR / "cross_family_family_summary.csv"),
        "common_candidates_csv": str(OUT_DIR / "cross_family_common_candidates.csv"),
        "scenario_only_common_csv": str(OUT_DIR / "cross_family_common_scenario_only_candidates.csv"),
        "ngl_only_never_selected_csv": str(OUT_DIR / "cross_family_never_selected_ngl_only.csv"),
        "counts_png": str(OUT_DIR / "01_cross_family_candidate_counts.png"),
        "heatmap_png": str(OUT_DIR / "02_cross_family_presence_heatmap.png"),
        "summary_md": str(OUT_DIR / "cross_family_comparison_summary.md"),
    }
    (OUT_DIR / "cross_family_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    df = load_scenario_candidates()
    build_outputs(df)


if __name__ == "__main__":
    main()

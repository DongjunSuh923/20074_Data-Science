from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
NEXT10_MUST_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_hubs" / "scenario_next10_excluding_must_have.csv"
NEXT10_BACKBONE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_backbone_only" / "scenario_next10_excluding_backbone_only.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "tier_a_hub_candidates"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))
    a = pd.read_csv(NEXT10_MUST_PATH)
    b = pd.read_csv(NEXT10_BACKBONE_PATH)

    a = a.loc[a["scenario_name"] != "baseline_no_shock"].copy()
    b = b.loc[b["scenario_name"] != "baseline_no_shock"].copy()

    a["in_must_have_exclusion"] = 1
    b["in_backbone_only_exclusion"] = 1

    # Only keep non-must-have states as actual expansion candidates.
    b_non_must = b.loc[~b["state_abbr"].isin(must_have)].copy()

    a_counts = a.groupby("state_abbr", as_index=False).agg(
        must_have_scenario_count=("scenario_name", "nunique"),
        must_have_best_rank=("next10_rank", "min"),
        must_have_avg_rank=("next10_rank", "mean"),
    )
    b_counts = b_non_must.groupby("state_abbr", as_index=False).agg(
        backbone_only_scenario_count=("scenario_name", "nunique"),
        backbone_only_best_rank=("next10_rank", "min"),
        backbone_only_avg_rank=("next10_rank", "mean"),
    )

    overlap = a[["scenario_name", "state_abbr", "next10_rank"]].merge(
        b_non_must[["scenario_name", "state_abbr", "next10_rank"]],
        on=["scenario_name", "state_abbr"],
        how="inner",
        suffixes=("_must", "_backbone"),
    )
    overlap_counts = overlap.groupby("state_abbr", as_index=False).agg(
        overlap_scenario_count=("scenario_name", "nunique"),
        overlap_best_combined_rank=("next10_rank_must", "min"),
    )

    merged = (
        a_counts.merge(b_counts, on="state_abbr", how="inner")
        .merge(overlap_counts, on="state_abbr", how="left")
        .fillna({"overlap_scenario_count": 0, "overlap_best_combined_rank": 999})
    )
    merged["overlap_scenario_count"] = merged["overlap_scenario_count"].astype(int)
    merged["tier_a_score"] = (
        2.0 * merged["overlap_scenario_count"]
        + 1.0 * merged["must_have_scenario_count"]
        + 0.5 * merged["backbone_only_scenario_count"]
        - 0.1 * merged["must_have_avg_rank"]
    )
    merged = merged.sort_values(
        ["overlap_scenario_count", "must_have_scenario_count", "backbone_only_scenario_count", "must_have_best_rank", "backbone_only_best_rank"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)

    # Tier A: repeated in both lenses and meaningful under must-have exclusion
    tier_a = merged.loc[
        (merged["overlap_scenario_count"] >= 2)
        & (merged["must_have_scenario_count"] >= 3)
        & (merged["backbone_only_scenario_count"] >= 2)
    ].copy()
    tier_a["tier_a_rank"] = range(1, len(tier_a) + 1)

    merged.to_csv(OUT_DIR / "tier_a_candidate_screening.csv", index=False)
    overlap.to_csv(OUT_DIR / "tier_a_overlap_by_scenario.csv", index=False)
    tier_a.to_csv(OUT_DIR / "tier_a_final_candidates.csv", index=False)

    lines = [
        "# Tier A Hub Candidate Rule",
        "",
        "Tier A is restricted to non-must-have states that survive both lenses:",
        "- must-have exclusion lens: still ranks inside next10 when the current core hubs are removed",
        "- backbone-only exclusion lens: still ranks inside next10 when only backbone hubs are removed",
        "",
        "Operational rule used here:",
        "- overlap_scenario_count >= 2",
        "- must_have_scenario_count >= 3",
        "- backbone_only_scenario_count >= 2",
        "",
        "Final Tier A candidates:",
    ]
    for row in tier_a.itertuples(index=False):
        lines.append(
            f"- {row.state_abbr}: overlap={row.overlap_scenario_count}, must_have_count={row.must_have_scenario_count}, "
            f"backbone_only_count={row.backbone_only_scenario_count}, best_must_rank={row.must_have_best_rank}, best_backbone_rank={row.backbone_only_best_rank}"
        )
    (OUT_DIR / "TIER_A_HUB_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

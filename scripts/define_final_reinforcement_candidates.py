from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

CROSS_DIR = WORK_ROOT / "outputs" / "scenario_model" / "cross_family_comparison"
BASELINE_ROUTES = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "final_reinforcement_candidates"

STATE_FIPS_TO_ABBR = {
    1: "AL", 4: "AZ", 5: "AR", 6: "CA", 8: "CO", 9: "CT", 10: "DE", 11: "DC", 12: "FL", 13: "GA",
    16: "ID", 17: "IL", 18: "IN", 19: "IA", 20: "KS", 21: "KY", 22: "LA", 23: "ME", 24: "MD", 25: "MA",
    26: "MI", 27: "MN", 28: "MS", 29: "MO", 30: "MT", 31: "NE", 32: "NV", 33: "NH", 34: "NJ", 35: "NM",
    36: "NY", 37: "NC", 38: "ND", 39: "OH", 40: "OK", 41: "OR", 42: "PA", 44: "RI", 45: "SC", 46: "SD",
    47: "TN", 48: "TX", 49: "UT", 50: "VT", 51: "VA", 53: "WA", 54: "WV", 55: "WI", 56: "WY",
}

SCTG_LABELS = {
    1: "Live animals/fish",
    2: "Cereal grains",
    3: "Other ag products",
    4: "Animal feed",
    5: "Meat/seafood",
    6: "Milled grain/food",
    7: "Other foodstuffs",
    8: "Alcohol/tobacco",
    9: "Building stone",
    10: "Natural sands",
    11: "Natural sands",
    12: "Gravel",
    13: "Nonmetallic minerals",
    14: "Metal ores",
    15: "Coal",
    16: "Crude petroleum",
    17: "Gasoline",
    18: "Fuel oils",
    19: "Coal-n.e.c./energy",
    20: "Basic chemicals",
    21: "Pharmaceuticals",
    22: "Fertilizers",
    23: "Chemical products",
    24: "Plastics/rubber",
    25: "Logs",
    26: "Wood products",
    27: "Pulp/newsprint",
    28: "Paper articles",
    29: "Printed products",
    30: "Textiles/leather",
    31: "Nonmetal mineral products",
    32: "Base metal",
    33: "Articles-base metal",
    34: "Machinery",
    35: "Electronics",
    36: "Motorized vehicles",
    37: "Transport equipment",
    38: "Precision instruments",
    39: "Furniture",
    40: "Misc manufacturing",
    41: "Waste/scrap",
    43: "Mixed freight",
}

NGL_CURRENT = {
    "AL","AZ","CA","CO","CT","FL","IA","IL","IN","KS","KY","LA","MA","MI","MN","MO","MS","NE","NV",
    "NY","OH","OK","OR","PA","RI","SD","TN","TX","WA","WI"
}


def classify_role(top_codes: list[int], state: str, family_count: int, is_ngl_current: bool) -> tuple[str, str]:
    codes = set(top_codes[:5])
    if state in {"NC", "VA", "NJ"}:
        return "East Coast corridor reinforcement", "High repeat survival plus eastern corridor and trade-linked distribution relevance"
    if state in {"WA", "OR", "LA"}:
        return "Gateway / trade redistribution support", "Pacific or Gulf gateway support with repeated survival under trade and capacity stress"
    if codes & {2, 3, 4} and state in {"MN", "NE", "KS", "ND", "SD", "IA"}:
        return "Grain / plains balancing support", "Strong agricultural/grain mix and repeated survival under grain, capacity, and demand shifts"
    if codes & {10, 11, 12, 31}:
        return "Construction / material balancing support", "Meaningful construction-material mix with repeated survival in domestic demand and capacity stress"
    if codes & {17, 18, 16}:
        return "Energy / fuel support", "Fuel-energy flows remain important in state touch profile"
    if family_count >= 4 and is_ngl_current:
        return "Portfolio internal reinforcement", "Repeated cross-family survival within current NGL footprint"
    return "General inland balancing support", "Diversified inland support role across multiple shock families"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cross = pd.read_csv(CROSS_DIR / "cross_family_candidate_matrix.csv")
    scenario_only = pd.read_csv(CROSS_DIR / "cross_family_common_scenario_only_candidates.csv")
    routes = pd.read_csv(BASELINE_ROUTES)
    routes = routes.loc[(routes["year"] == 2031.0) & (routes["scenario"] == "baseline")].copy()
    routes["orig_state"] = routes["orig_state_fips"].map(STATE_FIPS_TO_ABBR)
    routes["dest_state"] = routes["dest_state_fips"].map(STATE_FIPS_TO_ABBR)

    # Final reinforcement redefinition
    core_expansion = {"NC", "VA"}
    robust_internal = {"WA", "MN", "NE", "KS", "CO"}
    secondary_internal = {"OR", "WI", "AL", "LA"}
    conditional_watchlist = {"UT", "SC", "ND", "NM", "NJ", "SD", "NV"}

    final_rows = []
    for state in sorted(core_expansion | robust_internal | secondary_internal | conditional_watchlist):
        rec = cross.loc[cross["state_abbr"] == state].iloc[0]
        if state in core_expansion:
            tier = "Core expansion"
        elif state in robust_internal:
            tier = "Robust internal reinforcement"
        elif state in secondary_internal:
            tier = "Secondary internal reinforcement"
        else:
            tier = "Conditional watchlist"
        final_rows.append(
            {
                "state_abbr": state,
                "tier": tier,
                "is_ngl_current": bool(rec["is_ngl_current"]),
                "family_count": int(rec["family_count"]),
                "scenario_count_total": int(rec["scenario_count_total"]),
                "best_rank": int(rec["best_rank"]),
                "avg_rank": round(float(rec["avg_rank"]), 2),
            }
        )
    final_df = pd.DataFrame(final_rows).sort_values(
        ["tier", "family_count", "scenario_count_total", "best_rank"],
        ascending=[True, False, False, True],
    )
    final_df.to_csv(OUT_DIR / "final_reinforcement_candidate_set.csv", index=False)

    # Commodity role segmentation
    commodity_rows = []
    key_states = final_df["state_abbr"].tolist()
    for state in key_states:
        sub = routes.loc[(routes["orig_state"] == state) | (routes["dest_state"] == state)].copy()
        total = float(sub["prediction"].sum())
        agg = sub.groupby("sctg2", as_index=False)["prediction"].sum().sort_values("prediction", ascending=False)
        top5 = agg.head(5).copy()
        top_codes = top5["sctg2"].astype(int).tolist()
        role, rationale = classify_role(
            top_codes,
            state,
            int(cross.loc[cross["state_abbr"] == state, "family_count"].iloc[0]),
            bool(cross.loc[cross["state_abbr"] == state, "is_ngl_current"].iloc[0]),
        )
        for idx, row in top5.reset_index(drop=True).iterrows():
            share = float(row["prediction"]) / total if total else 0.0
            commodity_rows.append(
                {
                    "state_abbr": state,
                    "role": role,
                    "role_rationale": rationale,
                    "commodity_rank": idx + 1,
                    "sctg2": int(row["sctg2"]),
                    "commodity_label": SCTG_LABELS.get(int(row["sctg2"]), f"SCTG {int(row['sctg2'])}"),
                    "touch_2031": float(row["prediction"]),
                    "share_of_state_touch": round(share, 4),
                }
            )
    commodity_df = pd.DataFrame(commodity_rows)
    commodity_df.to_csv(OUT_DIR / "final_reinforcement_candidate_commodity_roles.csv", index=False)

    state_roles = (
        commodity_df.groupby(["state_abbr", "role", "role_rationale"], as_index=False)
        .agg(
            top1_commodity=("commodity_label", "first"),
            top1_share=("share_of_state_touch", "first"),
            top3_commodities=("commodity_label", lambda s: ", ".join(s.head(3))),
        )
        .merge(final_df, on="state_abbr", how="left")
        .sort_values(["tier", "family_count", "scenario_count_total", "best_rank"], ascending=[True, False, False, True])
    )
    state_roles.to_csv(OUT_DIR / "final_reinforcement_candidate_state_roles.csv", index=False)

    summary_lines = [
        "# Final Reinforcement Candidate Redefinition",
        "",
        "## Tier logic",
        "- Core expansion: non-NGL states with strongest cross-family repeated support",
        "- Robust internal reinforcement: current NGL states with strongest cross-family repeated support",
        "- Secondary internal reinforcement: repeated but one step below the robust set",
        "- Conditional watchlist: useful in multiple shocks, but more conditional or family-concentrated",
        "",
        "## Core expansion",
    ]
    for state in sorted(core_expansion):
        rec = state_roles.loc[state_roles["state_abbr"] == state].iloc[0]
        summary_lines.append(
            f"- {state}: {rec['role']}; top commodities: {rec['top3_commodities']}; families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}"
        )
    summary_lines.extend(["", "## Robust internal reinforcement"])
    for state in sorted(robust_internal):
        rec = state_roles.loc[state_roles["state_abbr"] == state].iloc[0]
        summary_lines.append(
            f"- {state}: {rec['role']}; top commodities: {rec['top3_commodities']}; families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}"
        )
    summary_lines.extend(["", "## Secondary internal reinforcement"])
    for state in sorted(secondary_internal):
        rec = state_roles.loc[state_roles["state_abbr"] == state].iloc[0]
        summary_lines.append(
            f"- {state}: {rec['role']}; top commodities: {rec['top3_commodities']}; families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}"
        )
    summary_lines.extend(["", "## Conditional watchlist"])
    for state in sorted(conditional_watchlist):
        rec = state_roles.loc[state_roles["state_abbr"] == state].iloc[0]
        summary_lines.append(
            f"- {state}: {rec['role']}; top commodities: {rec['top3_commodities']}; families={int(rec['family_count'])}, scenarios={int(rec['scenario_count_total'])}"
        )
    (OUT_DIR / "FINAL_REINFORCEMENT_CANDIDATE_REFERENCE.md").write_text("\n".join(summary_lines), encoding="utf-8")

    metadata = {
        "candidate_set_csv": str(OUT_DIR / "final_reinforcement_candidate_set.csv"),
        "commodity_roles_csv": str(OUT_DIR / "final_reinforcement_candidate_commodity_roles.csv"),
        "state_roles_csv": str(OUT_DIR / "final_reinforcement_candidate_state_roles.csv"),
        "reference_md": str(OUT_DIR / "FINAL_REINFORCEMENT_CANDIDATE_REFERENCE.md"),
    }
    (OUT_DIR / "final_reinforcement_candidate_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
BASELINE_RANK_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y" / "state_hub_rankings_2031.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_hubs"

STATE_ABBR_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
START_YEAR = 2026
SEVERITY_RULES = {
    "mild": {"start_loss": 0.05, "duration_years": 1},
    "medium": {"start_loss": 0.20, "duration_years": 3},
    "severe": {"start_loss": 0.40, "duration_years": 5},
}

SCENARIOS = [
    {
        "scenario_name": "snow_heat_state_blocking",
        "family": "Capacity shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "natural_disaster_state_blocking_cases.csv",
        "commodity_filter": None,
    },
    {
        "scenario_name": "tornado_state_blocking",
        "family": "Capacity shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "natural_disaster_state_blocking_cases.csv",
        "commodity_filter": None,
    },
    {
        "scenario_name": "wildfire_smoke_state_blocking",
        "family": "Capacity shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "natural_disaster_state_blocking_cases.csv",
        "commodity_filter": None,
    },
    {
        "scenario_name": "grain_supply_state_blocking",
        "family": "Commodity-specific supply shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock" / "grain_supply_state_blocking_cases.csv",
        "commodity_filter": {"2"},
    },
    {
        "scenario_name": "construction_infra_state_blocking",
        "family": "Demand shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "construction_infra_shock" / "construction_infra_state_blocking_cases.csv",
        "commodity_filter": {"11", "12", "31"},
    },
    {
        "scenario_name": "trade_external_state_blocking",
        "family": "Trade/external network shock",
        "case_path": PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "trade_external_state_blocking_cases.csv",
        "commodity_filter": None,
    },
]


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


def classify_hub_type(row: pd.Series) -> str:
    if row["interstate_share_2031"] >= 0.40:
        return "Backbone"
    return "Internal"


def load_cases(case_path: Path, scenario_name: str) -> pd.DataFrame:
    cases = pd.read_csv(case_path, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)
    if "scenario_name" in cases.columns:
        cases = cases.loc[cases["scenario_name"] == scenario_name].copy()
    return cases


def apply_state_blocking(
    routes: pd.DataFrame,
    cases: pd.DataFrame,
    commodity_filter: set[str] | None,
) -> pd.DataFrame:
    shocked = routes.copy()
    interstate_mask = shocked["orig_state_fips"] != shocked["dest_state_fips"]
    for row in cases.itertuples(index=False):
        state_mask = interstate_mask & (
            (shocked["orig_state_fips"] == row.state_fips) | (shocked["dest_state_fips"] == row.state_fips)
        )
        if commodity_filter is not None:
            state_mask = state_mask & shocked["sctg2"].isin(commodity_filter)
        for year, loss in schedule(row.severity).items():
            mask = state_mask & (shocked["year"] == year)
            shocked.loc[mask, "prediction"] = shocked.loc[mask, "prediction"] * (1 - loss)
    return shocked


def build_hub_metrics(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(pred_df["year"].dropna().astype(int).unique().tolist())
    states = sorted(set(pred_df["orig_state_fips"].dropna()) | set(pred_df["dest_state_fips"].dropna()))
    rows = []
    for year in years:
        year_df = pred_df.loc[pred_df["year"] == year].copy()
        for state in states:
            state_key = str(int(state)).zfill(2)
            out_inter = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "prediction"].sum()
            in_inter = year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "prediction"].sum()
            internal = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] == state), "prediction"].sum()
            touch_total = out_inter + in_inter + internal
            partners_out = set(year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "dest_state_fips"].astype(str))
            partners_in = set(year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "orig_state_fips"].astype(str))
            partner_count = len((partners_out | partners_in) - {"nan"})
            commodity_mask = (year_df["orig_state_fips"] == state) | (year_df["dest_state_fips"] == state)
            commodity_count = year_df.loc[commodity_mask & (year_df["prediction"] > 0), "sctg2"].nunique()
            internal_share = float(internal / touch_total) if touch_total > 0 else 0.0
            interstate_share = float((out_inter + in_inter) / touch_total) if touch_total > 0 else 0.0
            rows.append({
                "year": year,
                "state_fips": state,
                "state_abbr": STATE_ABBR_MAP.get(state_key, state_key),
                "touch_total": float(touch_total),
                "partner_count": int(partner_count),
                "commodity_count": int(commodity_count),
                "internal_share": internal_share,
                "interstate_share": interstate_share,
            })
    hub = pd.DataFrame(rows)
    base = hub.loc[hub["year"] == 2026, ["state_fips", "touch_total"]].rename(columns={"touch_total": "touch_2026"})
    last = hub.loc[hub["year"] == 2031, ["state_fips", "touch_total", "partner_count", "commodity_count", "internal_share", "interstate_share"]].rename(
        columns={
            "touch_total": "touch_2031",
            "partner_count": "partner_count_2031",
            "commodity_count": "commodity_count_2031",
            "internal_share": "internal_share_2031",
            "interstate_share": "interstate_share_2031",
        }
    )
    avg = hub.groupby(["state_fips", "state_abbr"], as_index=False)["touch_total"].mean().rename(columns={"touch_total": "avg_touch_2025_2031"})
    summary = avg.merge(base, on="state_fips", how="left").merge(last, on="state_fips", how="left")
    summary["touch_cagr_2026_2031"] = np.where(summary["touch_2026"] > 0, (summary["touch_2031"] / summary["touch_2026"]) ** (1 / 5) - 1, np.nan)
    summary["volume_rank_pct"] = summary["touch_2031"].rank(pct=True)
    summary["growth_rank_pct"] = summary["touch_cagr_2026_2031"].rank(pct=True)
    summary["partner_rank_pct"] = summary["partner_count_2031"].rank(pct=True)
    summary["commodity_rank_pct"] = summary["commodity_count_2031"].rank(pct=True)
    summary["hub_score"] = summary[["volume_rank_pct", "growth_rank_pct", "partner_rank_pct", "commodity_rank_pct"]].mean(axis=1) * 100.0
    summary = summary.sort_values(["hub_score", "touch_2031"], ascending=[False, False]).reset_index(drop=True)
    summary["hub_rank_2031"] = np.arange(1, len(summary) + 1)
    summary["hub_type"] = summary.apply(classify_hub_type, axis=1)
    return hub, summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["sctg2"] = routes["sctg2"].astype(str)

    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))
    baseline_rank = pd.read_csv(BASELINE_RANK_PATH)
    baseline_rank["hub_type"] = baseline_rank.apply(classify_hub_type, axis=1)
    baseline_next10 = baseline_rank.loc[~baseline_rank["state_abbr"].isin(must_have)].head(10).copy()
    baseline_next10["scenario_name"] = "baseline_no_shock"

    all_next10_rows = [baseline_next10.assign(next10_rank=np.arange(1, len(baseline_next10) + 1))]
    all_rank_rows = []

    for spec in SCENARIOS:
        cases = load_cases(spec["case_path"], spec["scenario_name"])
        shocked = apply_state_blocking(routes, cases, spec["commodity_filter"])
        shocked_path = OUT_DIR / f"{spec['scenario_name']}_shocked_routes_2025_2031.csv"
        shocked.to_csv(shocked_path, index=False)

        _, summary = build_hub_metrics(shocked)
        summary["scenario_name"] = spec["scenario_name"]
        summary["shock_family"] = spec["family"]
        summary.to_csv(OUT_DIR / f"{spec['scenario_name']}_hub_rankings_2031.csv", index=False)
        all_rank_rows.append(summary)

        next10 = summary.loc[~summary["state_abbr"].isin(must_have)].head(10).copy()
        next10["next10_rank"] = np.arange(1, len(next10) + 1)
        next10.to_csv(OUT_DIR / f"{spec['scenario_name']}_next10_excluding_must_have.csv", index=False)
        all_next10_rows.append(next10)

    combined_next10 = pd.concat(all_next10_rows, ignore_index=True, sort=False)
    combined_next10.to_csv(OUT_DIR / "scenario_next10_excluding_must_have.csv", index=False)

    combined_ranks = pd.concat(all_rank_rows, ignore_index=True, sort=False)
    combined_ranks.to_csv(OUT_DIR / "scenario_all_hub_rankings_2031.csv", index=False)

    comparison_rows = []
    baseline_lookup = baseline_rank.set_index("state_abbr")
    for scenario_name in combined_next10["scenario_name"].dropna().unique():
        if scenario_name == "baseline_no_shock":
            continue
        sub = combined_next10.loc[combined_next10["scenario_name"] == scenario_name].copy()
        for row in sub.itertuples(index=False):
            baseline_rank_num = int(baseline_lookup.loc[row.state_abbr, "hub_rank_2031"])
            baseline_hub_score = float(baseline_lookup.loc[row.state_abbr, "hub_score"])
            comparison_rows.append({
                "scenario_name": scenario_name,
                "state_abbr": row.state_abbr,
                "next10_rank": int(row.next10_rank),
                "scenario_hub_rank_2031": int(row.hub_rank_2031),
                "baseline_hub_rank_2031": baseline_rank_num,
                "rank_shift_vs_baseline": baseline_rank_num - int(row.hub_rank_2031),
                "scenario_hub_score": float(row.hub_score),
                "baseline_hub_score": baseline_hub_score,
                "hub_type": row.hub_type,
            })
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "scenario_next10_rank_shift_vs_baseline.csv", index=False)

    lines = [
        "# Scenario Next 10 Hub Ranking (Must-Have Fixed)",
        "",
        "Must-have hubs fixed and excluded from the candidate ranking:",
        f"- {', '.join(sorted(must_have))}",
        "",
        "Baseline next 10 after excluding must-have:",
        "- " + ", ".join(baseline_next10["state_abbr"].tolist()),
        "",
    ]
    for scenario_name in [s["scenario_name"] for s in SCENARIOS]:
        sub = combined_next10.loc[combined_next10["scenario_name"] == scenario_name]
        lines.append(f"## {scenario_name}")
        lines.append("- " + ", ".join(sub["state_abbr"].tolist()))
        leaders = comparison.loc[comparison["scenario_name"] == scenario_name].sort_values(["rank_shift_vs_baseline", "next10_rank"], ascending=[False, True]).head(3)
        if not leaders.empty:
            lines.append("Promoted vs baseline:")
            for row in leaders.itertuples(index=False):
                lines.append(f"- {row.state_abbr}: baseline_rank={row.baseline_hub_rank_2031}, scenario_rank={row.scenario_hub_rank_2031}, shift={row.rank_shift_vs_baseline:+d}, type={row.hub_type}")
        lines.append("")

    (OUT_DIR / "scenario_next10_summary.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "baseline_next10": str(OUT_DIR / "scenario_next10_excluding_must_have.csv"),
        "all_rankings": str(OUT_DIR / "scenario_all_hub_rankings_2031.csv"),
        "rank_shift": str(OUT_DIR / "scenario_next10_rank_shift_vs_baseline.csv"),
        "summary": str(OUT_DIR / "scenario_next10_summary.md"),
    }
    (OUT_DIR / "scenario_next10_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

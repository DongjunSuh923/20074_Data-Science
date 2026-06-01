from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
BASELINE_RANK_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y" / "state_hub_rankings_2031.csv"
BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
CRITICALITY_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "state_blocking_criticality_severe.csv"
SCENARIO_NEXT10_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_hubs"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next10_backbone_only"

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


def classify_backbone_members(base_rank: pd.DataFrame) -> pd.DataFrame:
    top20 = base_rank.loc[base_rank["hub_rank_2031"] <= 20].copy()
    top20["is_backbone"] = top20["interstate_share_2031"] >= 0.40
    return top20


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


def load_cases(case_path: Path, scenario_name: str) -> pd.DataFrame:
    cases = pd.read_csv(case_path, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)
    if "scenario_name" in cases.columns:
        cases = cases.loc[cases["scenario_name"] == scenario_name].copy()
    return cases


def apply_state_blocking(routes: pd.DataFrame, cases: pd.DataFrame, commodity_filter: set[str] | None) -> pd.DataFrame:
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


def build_hub_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
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
    summary["hub_type"] = np.where(summary["interstate_share_2031"] >= 0.40, "Backbone", "Internal")
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_rank = pd.read_csv(BASELINE_RANK_PATH)
    backbone_df = classify_backbone_members(base_rank)
    backbone_only = backbone_df.loc[backbone_df["is_backbone"], ["state_abbr", "hub_rank_2031", "interstate_share_2031", "internal_share_2031", "touch_2031"]].copy()

    crit = pd.read_csv(CRITICALITY_PATH)
    if "state_abbr" not in crit.columns or crit["state_abbr"].isna().all():
        crit["state_abbr"] = crit["blocked_state"]
    crit["cumulative_network_loss_pct_of_annual_sum"] = pd.to_numeric(crit["cumulative_network_loss_pct_of_annual_sum"], errors="coerce")
    backbone_validation = backbone_only.merge(
        crit[["state_abbr", "cumulative_network_loss_pct_of_annual_sum"]].rename(columns={"cumulative_network_loss_pct_of_annual_sum": "severe_cum_loss"}),
        on="state_abbr",
        how="left",
    )
    backbone_only.to_csv(OUT_DIR / "backbone_only_exclusion_set.csv", index=False)
    backbone_validation.to_csv(OUT_DIR / "backbone_only_validation.csv", index=False)

    summary_stats = pd.DataFrame(
        [
            {
                "group_name": "backbone_only",
                "count": len(backbone_validation),
                "avg_severe_cum_loss": backbone_validation["severe_cum_loss"].mean(),
                "max_severe_cum_loss": backbone_validation["severe_cum_loss"].max(),
                "min_baseline_rank": backbone_validation["hub_rank_2031"].min(),
                "max_baseline_rank": backbone_validation["hub_rank_2031"].max(),
            },
            {
                "group_name": "all_internal_top20",
                "count": int((~backbone_df["is_backbone"]).sum()),
                "avg_severe_cum_loss": crit.loc[crit["state_abbr"].isin(backbone_df.loc[~backbone_df["is_backbone"], "state_abbr"]), "cumulative_network_loss_pct_of_annual_sum"].mean(),
                "max_severe_cum_loss": crit.loc[crit["state_abbr"].isin(backbone_df.loc[~backbone_df["is_backbone"], "state_abbr"]), "cumulative_network_loss_pct_of_annual_sum"].max(),
                "min_baseline_rank": backbone_df.loc[~backbone_df["is_backbone"], "hub_rank_2031"].min(),
                "max_baseline_rank": backbone_df.loc[~backbone_df["is_backbone"], "hub_rank_2031"].max(),
            },
        ]
    )
    summary_stats.to_csv(OUT_DIR / "backbone_importance_summary.csv", index=False)

    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["sctg2"] = routes["sctg2"].astype(str)

    baseline_next10 = base_rank.loc[~base_rank["state_abbr"].isin(set(backbone_only["state_abbr"]))].head(10).copy()
    baseline_next10["scenario_name"] = "baseline_no_shock"
    baseline_next10["next10_rank"] = np.arange(1, len(baseline_next10) + 1)
    all_next10 = [baseline_next10]

    for spec in SCENARIOS:
        cases = load_cases(spec["case_path"], spec["scenario_name"])
        shocked = apply_state_blocking(routes, cases, spec["commodity_filter"])
        summary = build_hub_metrics(shocked)
        summary["scenario_name"] = spec["scenario_name"]
        next10 = summary.loc[~summary["state_abbr"].isin(set(backbone_only["state_abbr"]))].head(10).copy()
        next10["next10_rank"] = np.arange(1, len(next10) + 1)
        next10.to_csv(OUT_DIR / f"{spec['scenario_name']}_next10_excluding_backbone_only.csv", index=False)
        all_next10.append(next10)

    combined = pd.concat(all_next10, ignore_index=True, sort=False)
    combined.to_csv(OUT_DIR / "scenario_next10_excluding_backbone_only.csv", index=False)

    lines = [
        "# Backbone-only Exclusion Validation",
        "",
        "Backbone-only exclusion set is defined from baseline 2031 top20 states with `interstate_share_2031 >= 0.40`.",
        "",
        "Selected backbone-only exclusion set:",
        "- " + ", ".join(backbone_only["state_abbr"].tolist()),
        "",
        "Validation logic:",
        "- these states should not only be backbone-oriented by mix",
        "- they should also carry meaningful severe state-blocking criticality",
        "",
        f"- average severe cumulative loss (backbone-only set): {backbone_validation['severe_cum_loss'].mean():.2%}",
        f"- max severe cumulative loss (backbone-only set): {backbone_validation['severe_cum_loss'].max():.2%}",
        "",
        "Baseline next 10 after excluding backbone-only set:",
        "- " + ", ".join(baseline_next10["state_abbr"].tolist()),
    ]
    (OUT_DIR / "BACKBONE_ONLY_EXCLUSION_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "backbone_only_metadata.json").write_text(
        json.dumps(
            {
                "backbone_only_exclusion_set": str(OUT_DIR / "backbone_only_exclusion_set.csv"),
                "backbone_validation": str(OUT_DIR / "backbone_only_validation.csv"),
                "importance_summary": str(OUT_DIR / "backbone_importance_summary.csv"),
                "scenario_next10": str(OUT_DIR / "scenario_next10_excluding_backbone_only.csv"),
                "reference": str(OUT_DIR / "BACKBONE_ONLY_EXCLUSION_REFERENCE.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

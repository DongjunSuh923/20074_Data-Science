from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
MUST_HAVE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
CONSTRUCTION_CASES_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "construction_infra_shock" / "construction_infra_state_blocking_cases.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "domestic_issue_hybrid"
START_YEAR = 2026
TOP_N = 15

SEVERITY_RULES = {
    "mild": {"start_loss": 0.05, "duration_years": 1},
    "medium": {"start_loss": 0.20, "duration_years": 3},
    "severe": {"start_loss": 0.40, "duration_years": 5},
}

STATE_ABBR_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
CONSTRUCTION_COMMODITIES = {"11", "12", "31"}


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


def load_routes() -> pd.DataFrame:
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["sctg2"] = routes["sctg2"].astype(str)
    return routes


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(CONSTRUCTION_CASES_PATH, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)
    return cases


def apply_state_blocking(shocked: pd.DataFrame, cases: pd.DataFrame) -> None:
    interstate_mask = shocked["orig_state_fips"] != shocked["dest_state_fips"]
    for row in cases.itertuples(index=False):
        state_mask = interstate_mask & (
            (shocked["orig_state_fips"] == row.state_fips) | (shocked["dest_state_fips"] == row.state_fips)
        ) & shocked["sctg2"].isin(CONSTRUCTION_COMMODITIES)
        for year, loss in schedule(row.severity).items():
            mask = state_mask & (shocked["year"] == year)
            shocked.loc[mask, "prediction"] = shocked.loc[mask, "prediction"] * (1 - loss)
            shocked.loc[mask, "state_block_loss_applied"] = shocked.loc[mask, "state_block_loss_applied"] + loss


def apply_destination_overlay(shocked: pd.DataFrame, cases: pd.DataFrame, overlay_scale: float = 0.5) -> None:
    for row in cases.itertuples(index=False):
        state_mask = (shocked["dest_state_fips"] == row.state_fips) & shocked["sctg2"].isin(CONSTRUCTION_COMMODITIES)
        for year, loss in schedule(row.severity).items():
            overlay_loss = loss * overlay_scale
            mask = state_mask & (shocked["year"] == year)
            shocked.loc[mask, "prediction"] = shocked.loc[mask, "prediction"] * (1 - overlay_loss)
            shocked.loc[mask, "overlay_loss_applied"] = shocked.loc[mask, "overlay_loss_applied"] + overlay_loss


def build_hub_rankings(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].unique().tolist())
    states = sorted(set(df["orig_state_fips"]) | set(df["dest_state_fips"]))
    rows = []
    for year in years:
        year_df = df.loc[df["year"] == year]
        for state in states:
            state_key = str(int(state)).zfill(2)
            out_inter = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "prediction"].sum()
            in_inter = year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "prediction"].sum()
            internal = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] == state), "prediction"].sum()
            touch_total = out_inter + in_inter + internal
            partner_count = len(
                set(year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "dest_state_fips"].astype(str))
                | set(year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "orig_state_fips"].astype(str))
            )
            commodity_mask = (year_df["orig_state_fips"] == state) | (year_df["dest_state_fips"] == state)
            commodity_count = year_df.loc[commodity_mask & (year_df["prediction"] > 0), "sctg2"].nunique()
            rows.append(
                {
                    "year": year,
                    "state_fips": state,
                    "state_abbr": STATE_ABBR_MAP.get(state_key, state_key),
                    "touch_total": float(touch_total),
                    "partner_count": int(partner_count),
                    "commodity_count": int(commodity_count),
                    "internal_share": float(internal / touch_total) if touch_total > 0 else 0.0,
                    "interstate_share": float((out_inter + in_inter) / touch_total) if touch_total > 0 else 0.0,
                }
            )
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
    summary["touch_cagr_2026_2031"] = (summary["touch_2031"] / summary["touch_2026"]).pow(1 / 5) - 1
    for col_in, col_out in [
        ("touch_2031", "volume_rank_pct"),
        ("touch_cagr_2026_2031", "growth_rank_pct"),
        ("partner_count_2031", "partner_rank_pct"),
        ("commodity_count_2031", "commodity_rank_pct"),
    ]:
        summary[col_out] = summary[col_in].rank(pct=True)
    summary["hub_score"] = summary[["volume_rank_pct", "growth_rank_pct", "partner_rank_pct", "commodity_rank_pct"]].mean(axis=1) * 100.0
    summary = summary.sort_values(["hub_score", "touch_2031"], ascending=[False, False]).reset_index(drop=True)
    summary["hub_rank_2031"] = range(1, len(summary) + 1)
    summary["hub_type"] = summary["interstate_share_2031"].ge(0.40).map({True: "Backbone", False: "Internal"})
    return summary


def summarize_losses(base: pd.DataFrame, shocked: pd.DataFrame) -> dict:
    annual_base = base.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_total"})
    annual_shocked = shocked.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shocked_total"})
    annual = annual_base.merge(annual_shocked, on="year", how="left")
    annual["loss_pct"] = (annual["base_total"] - annual["shocked_total"]) / annual["base_total"]

    subset_base = base.loc[base["sctg2"].isin(CONSTRUCTION_COMMODITIES)].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_subset"})
    subset_shocked = shocked.loc[shocked["sctg2"].isin(CONSTRUCTION_COMMODITIES)].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shocked_subset"})
    subset = subset_base.merge(subset_shocked, on="year", how="left")
    subset["loss_pct"] = (subset["base_subset"] - subset["shocked_subset"]) / subset["base_subset"]

    return {
        "scenario_name": "construction_infra_hybrid",
        "cumulative_network_loss_pct_of_annual_sum": annual["loss_pct"].sum(),
        "peak_network_loss_pct": annual["loss_pct"].max(),
        "peak_network_loss_year": int(annual.loc[annual["loss_pct"].idxmax(), "year"]),
        "cumulative_subset_loss_pct_of_annual_sum": subset["loss_pct"].sum(),
        "peak_subset_loss_pct": subset["loss_pct"].max(),
        "peak_subset_loss_year": int(subset.loc[subset["loss_pct"].idxmax(), "year"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = load_routes()
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))
    cases = load_cases()

    shocked = routes.copy()
    shocked["state_block_loss_applied"] = 0.0
    shocked["overlay_loss_applied"] = 0.0
    apply_state_blocking(shocked, cases)
    apply_destination_overlay(shocked, cases, overlay_scale=0.5)

    rankings = build_hub_rankings(shocked)
    next15 = rankings.loc[~rankings["state_abbr"].isin(must_have)].head(TOP_N).copy()
    next15["next_rank"] = range(1, len(next15) + 1)
    losses = summarize_losses(routes, shocked)

    shocked.to_csv(OUT_DIR / "construction_infra_hybrid_shocked_routes_2025_2031.csv", index=False)
    rankings.to_csv(OUT_DIR / "construction_infra_hybrid_hub_rankings_2031.csv", index=False)
    next15.to_csv(OUT_DIR / "construction_infra_hybrid_next15_excluding_must_have.csv", index=False)
    pd.DataFrame([losses]).to_csv(OUT_DIR / "construction_infra_hybrid_summary.csv", index=False)

    lines = [
        "# Domestic Issue Hybrid Reference",
        "",
        "This track is separated from natural-disaster scenarios.",
        "Current hybrid implementation:",
        "- state blocking on construction-linked interstate flows",
        "- destination-side demand overlay on construction-linked commodities",
        "- same-state short-haul remains excluded",
        "",
        f"- cumulative network loss: {losses['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {losses['peak_network_loss_pct']:.2%} in {losses['peak_network_loss_year']}",
        f"- cumulative construction subset loss: {losses['cumulative_subset_loss_pct_of_annual_sum']:.2%}",
        f"- peak construction subset loss: {losses['peak_subset_loss_pct']:.2%} in {losses['peak_subset_loss_year']}",
        "",
        "Top 15 non-must-have candidates under this domestic issue hybrid:",
        "- " + ", ".join(next15["state_abbr"].tolist()),
    ]
    (OUT_DIR / "DOMESTIC_ISSUE_HYBRID_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "domestic_issue_hybrid_metadata.json").write_text(
        json.dumps(
            {
                "summary_csv": str(OUT_DIR / "construction_infra_hybrid_summary.csv"),
                "next15_csv": str(OUT_DIR / "construction_infra_hybrid_next15_excluding_must_have.csv"),
                "rankings_csv": str(OUT_DIR / "construction_infra_hybrid_hub_rankings_2031.csv"),
                "reference_md": str(OUT_DIR / "DOMESTIC_ISSUE_HYBRID_REFERENCE.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

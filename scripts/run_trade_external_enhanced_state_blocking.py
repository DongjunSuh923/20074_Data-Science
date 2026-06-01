from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


WORK_ROOT = Path(__file__).resolve().parents[1]
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
BASE_ROUTE_PATH = IDEA_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
CASE_PATH = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_enhanced_shock" / "trade_external_enhanced_state_blocking_cases.csv"
MUST_HAVE_PATH = IDEA_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_enhanced_shock" / "simulation_results"
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


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["sctg2"] = routes["sctg2"].astype(str)
    cases = pd.read_csv(CASE_PATH, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))

    shocked = routes.copy()
    interstate_mask = shocked["orig_state_fips"] != shocked["dest_state_fips"]
    for row in cases.itertuples(index=False):
        state_mask = interstate_mask & (
            (shocked["orig_state_fips"] == row.state_fips) | (shocked["dest_state_fips"] == row.state_fips)
        )
        for year, loss in schedule(row.severity).items():
            mask = state_mask & (shocked["year"] == year)
            shocked.loc[mask, "prediction"] = shocked.loc[mask, "prediction"] * (1 - loss)

    yearly = routes.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_total"})
    yearly = yearly.merge(shocked.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shock_total"}), on="year", how="left")
    yearly["loss_abs"] = yearly["base_total"] - yearly["shock_total"]
    yearly["loss_pct"] = yearly["loss_abs"] / yearly["base_total"]

    interstate_base = routes.loc[interstate_mask].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_interstate_total"})
    interstate_shock = shocked.loc[interstate_mask].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shock_interstate_total"})
    interstate = interstate_base.merge(interstate_shock, on="year", how="left")
    interstate["loss_abs"] = interstate["base_interstate_total"] - interstate["shock_interstate_total"]
    interstate["loss_pct"] = interstate["loss_abs"] / interstate["base_interstate_total"]

    rankings = build_hub_rankings(shocked)
    next15 = rankings.loc[~rankings["state_abbr"].isin(must_have)].head(TOP_N).copy()
    next15["next_rank"] = range(1, len(next15) + 1)

    summary = {
        "scenario_name": "trade_external_enhanced_state_blocking",
        "blocked_states": ",".join(cases["state_fips"].tolist()),
        "cumulative_network_loss_pct_of_annual_sum": float(yearly["loss_abs"].sum() / yearly["base_total"].sum()),
        "peak_network_loss_pct": float(yearly["loss_pct"].max()),
        "peak_network_loss_year": int(yearly.loc[yearly["loss_pct"].idxmax(), "year"]),
        "cumulative_interstate_loss_pct_of_annual_sum": float(interstate["loss_abs"].sum() / interstate["base_interstate_total"].sum()),
        "peak_interstate_loss_pct": float(interstate["loss_pct"].max()),
        "peak_interstate_loss_year": int(interstate.loc[interstate["loss_pct"].idxmax(), "year"]),
    }

    yearly.to_csv(OUT_DIR / "trade_external_enhanced_network_totals.csv", index=False)
    interstate.to_csv(OUT_DIR / "trade_external_enhanced_interstate_totals.csv", index=False)
    rankings.to_csv(OUT_DIR / "trade_external_enhanced_hub_rankings_2031.csv", index=False)
    next15.to_csv(OUT_DIR / "trade_external_enhanced_next15_excluding_must_have.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUT_DIR / "trade_external_enhanced_summary.csv", index=False)
    (OUT_DIR / "trade_external_enhanced_summary.md").write_text(
        "\n".join(
            [
                "# Enhanced Trade / External Network Shock Summary",
                "",
                f"- blocked states: {', '.join(cases['state_abbr'].tolist())}",
                f"- cumulative network loss: {summary['cumulative_network_loss_pct_of_annual_sum']:.2%}",
                f"- peak network loss: {summary['peak_network_loss_pct']:.2%} in {summary['peak_network_loss_year']}",
                f"- cumulative interstate loss: {summary['cumulative_interstate_loss_pct_of_annual_sum']:.2%}",
                f"- peak interstate loss: {summary['peak_interstate_loss_pct']:.2%} in {summary['peak_interstate_loss_year']}",
                "",
                "Top 15 non-must-have candidates:",
                "- " + ", ".join(next15["state_abbr"].tolist()),
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "trade_external_enhanced_metadata.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "next15_csv": str(OUT_DIR / "trade_external_enhanced_next15_excluding_must_have.csv"),
                "rankings_csv": str(OUT_DIR / "trade_external_enhanced_hub_rankings_2031.csv"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
CASE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "trade_external_state_blocking_cases.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock" / "simulation_results"

START_YEAR = 2026
SEVERITY_RULES = {
    "mild": {"start_loss": 0.05, "duration_years": 1},
    "medium": {"start_loss": 0.20, "duration_years": 3},
    "severe": {"start_loss": 0.40, "duration_years": 5},
}


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    cases = pd.read_csv(CASE_PATH, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)

    shocked = routes.copy()
    manifest = []
    interstate_mask = shocked["orig_state_fips"] != shocked["dest_state_fips"]
    for row in cases.itertuples(index=False):
        sc = schedule(row.severity)
        state_mask = interstate_mask & (
            (shocked["orig_state_fips"] == row.state_fips) | (shocked["dest_state_fips"] == row.state_fips)
        )
        for year, loss in sc.items():
            year_mask = state_mask & (shocked["year"] == year)
            shocked.loc[year_mask, "prediction"] = shocked.loc[year_mask, "prediction"] * (1 - loss)
        manifest.append({"state_fips": row.state_fips, "state_abbr": row.state_abbr, "severity": row.severity, "schedule": sc})

    base_total = routes.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_total"})
    shock_total = shocked.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shock_total"})
    yearly = base_total.merge(shock_total, on="year", how="left")
    yearly["loss_abs"] = yearly["base_total"] - yearly["shock_total"]
    yearly["loss_pct"] = yearly["loss_abs"] / yearly["base_total"]
    yearly.to_csv(OUT_DIR / "trade_external_network_totals.csv", index=False)

    interstate_base = routes.loc[interstate_mask].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_interstate_total"})
    interstate_shock = shocked.loc[interstate_mask].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shock_interstate_total"})
    interstate = interstate_base.merge(interstate_shock, on="year", how="left")
    interstate["loss_abs"] = interstate["base_interstate_total"] - interstate["shock_interstate_total"]
    interstate["loss_pct"] = interstate["loss_abs"] / interstate["base_interstate_total"]
    interstate.to_csv(OUT_DIR / "trade_external_interstate_totals.csv", index=False)

    summary = {
        "scenario_name": "trade_external_state_blocking",
        "blocked_states": list(cases["state_fips"]),
        "cumulative_network_loss_pct_of_annual_sum": float(yearly["loss_abs"].sum() / yearly["base_total"].sum()),
        "peak_year_loss_pct": float(yearly["loss_pct"].max()),
        "peak_year": int(yearly.loc[yearly["loss_pct"].idxmax(), "year"]),
        "interstate_cumulative_loss_pct_of_annual_sum": float(interstate["loss_abs"].sum() / interstate["base_interstate_total"].sum()),
        "interstate_peak_year_loss_pct": float(interstate["loss_pct"].max()),
    }
    pd.DataFrame([summary]).to_csv(OUT_DIR / "trade_external_simulation_summary.csv", index=False)
    (OUT_DIR / "trade_external_simulation_manifest.json").write_text(json.dumps({"states": manifest, "summary": summary}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Trade / External Network State Blocking Simulation Summary",
        "",
        f"- blocked state_fips: {', '.join(summary['blocked_states'])}",
        f"- cumulative network loss: {summary['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {summary['peak_year_loss_pct']:.2%} in {summary['peak_year']}",
        f"- interstate cumulative loss: {summary['interstate_cumulative_loss_pct_of_annual_sum']:.2%}",
        f"- interstate peak loss: {summary['interstate_peak_year_loss_pct']:.2%}",
        "",
        "Method note:",
        "- same-state short-haul is excluded from the shock mask by design",
    ]
    (OUT_DIR / "trade_external_simulation_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

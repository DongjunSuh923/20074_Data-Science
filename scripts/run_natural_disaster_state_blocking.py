from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
BASE_RANK_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_state_hub_rankings_2031.csv"
CASE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "natural_disaster_state_blocking_cases.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "simulation_results"

START_YEAR = 2026
SEVERITY_RULES = {
    "mild": {"start_loss": 0.05, "duration_years": 1},
    "medium": {"start_loss": 0.20, "duration_years": 3},
    "severe": {"start_loss": 0.40, "duration_years": 5},
}


def build_schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    schedule: dict[int, float] = {}
    for offset in range(rule["duration_years"]):
        schedule[START_YEAR + offset] = loss
        loss /= 2
    return schedule


def classify_hub_type(interstate_share: float) -> str:
    return "Backbone" if interstate_share >= 0.40 else "Internal"


def apply_case(routes: pd.DataFrame, blocked_states: list[tuple[str, str]]) -> tuple[pd.DataFrame, dict[str, dict[int, float]]]:
    shocked = routes.copy()
    schedules: dict[str, dict[int, float]] = {}
    for state_fips, severity in blocked_states:
        schedule = build_schedule(severity)
        schedules[state_fips] = schedule
        mask = (shocked["orig_state_fips"] == state_fips) | (shocked["dest_state_fips"] == state_fips)
        for year, loss in schedule.items():
            year_mask = mask & (shocked["year"] == year)
            shocked.loc[year_mask, "prediction"] = shocked.loc[year_mask, "prediction"] * (1 - loss)
    return shocked, schedules


def summarize_case(name: str, base_routes: pd.DataFrame, shocked_routes: pd.DataFrame, blocked_states: list[tuple[str, str]], schedules: dict[str, dict[int, float]], rank_df: pd.DataFrame) -> dict:
    base_total = base_routes.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_total"})
    shock_total = shocked_routes.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shock_total"})
    yearly = base_total.merge(shock_total, on="year", how="left")
    yearly["loss_abs"] = yearly["base_total"] - yearly["shock_total"]
    yearly["loss_pct"] = yearly["loss_abs"] / yearly["base_total"]
    annual_path = OUT_DIR / f"{name}_network_totals.csv"
    yearly.to_csv(annual_path, index=False)

    blocked_details = []
    for state_fips, severity in blocked_states:
        row = rank_df.loc[rank_df["state_fips"] == state_fips].iloc[0]
        blocked_details.append(
            {
                "state_fips": state_fips,
                "state_abbr": row["state_abbr"],
                "hub_rank_2031": int(row["hub_rank_2031"]),
                "hub_type": classify_hub_type(float(row["interstate_share_2031"])),
                "severity": severity,
                "schedule": schedules[state_fips],
            }
        )

    return {
        "scenario_name": name,
        "blocked_states": blocked_details,
        "cumulative_network_loss_pct_of_annual_sum": float(yearly["loss_abs"].sum() / yearly["base_total"].sum()),
        "peak_year_loss_pct": float(yearly["loss_pct"].max()),
        "peak_year": int(yearly.loc[yearly["loss_pct"].idxmax(), "year"]),
        "output_file": str(annual_path),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes = pd.read_csv(
        BASE_ROUTE_PATH,
        dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string", "dist_band": "string", "trade_type": "string"},
    )
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    rank_df = pd.read_csv(BASE_RANK_PATH, dtype={"state_fips": "string", "state_abbr": "string"})
    rank_df["state_fips"] = rank_df["state_fips"].astype(str).str.zfill(2)
    cases = pd.read_csv(CASE_PATH, dtype={"state_fips": "string"})
    cases["state_fips"] = cases["state_fips"].astype(str).str.zfill(2)

    summaries = []
    for scenario_name, grp in cases.groupby("scenario_name"):
        blocked_states = list(zip(grp["state_fips"], grp["severity"]))
        shocked, schedules = apply_case(routes, blocked_states)
        summaries.append(summarize_case(scenario_name, routes, shocked, blocked_states, schedules, rank_df))

    summary_df = pd.DataFrame(
        [
            {
                "scenario_name": s["scenario_name"],
                "blocked_state_count": len(s["blocked_states"]),
                "blocked_states": ",".join(x["state_abbr"] for x in s["blocked_states"]),
                "cumulative_network_loss_pct_of_annual_sum": s["cumulative_network_loss_pct_of_annual_sum"],
                "peak_year_loss_pct": s["peak_year_loss_pct"],
                "peak_year": s["peak_year"],
            }
            for s in summaries
        ]
    ).sort_values("cumulative_network_loss_pct_of_annual_sum", ascending=False)
    summary_df.to_csv(OUT_DIR / "natural_disaster_simulation_summary.csv", index=False)
    (OUT_DIR / "natural_disaster_simulation_manifest.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# Natural Disaster State Blocking Simulation Summary", ""]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"- {row.scenario_name}: blocked={row.blocked_states}, cumulative_loss={row.cumulative_network_loss_pct_of_annual_sum:.2%}, "
            f"peak_year={row.peak_year}, peak_loss={row.peak_year_loss_pct:.2%}"
        )
    (OUT_DIR / "natural_disaster_simulation_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

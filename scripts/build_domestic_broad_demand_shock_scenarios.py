from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WORK_ROOT = Path(__file__).resolve().parents[1]
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
SAPCE_DIR = WORK_ROOT / "tmp_SAPCE"
SAS_DIR = WORK_ROOT / "tmp_SASUMMARY"
BASE_ROUTE_PATH = IDEA_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "domestic_broad_demand_shock"

SEVERITY_BY_RANK = {1: "severe", 2: "medium", 3: "medium", 4: "mild", 5: "mild"}
STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}
EXCLUDED = {"AK", "HI", "DC"}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def load_sas_metrics(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    row = lambda code: df.loc[df["LineCode"] == code].iloc[0]
    years = ["2021", "2022", "2023", "2024"]
    real_gdp = row(1)
    real_pce = row(3)
    employment = row(15)

    def cagr(r: pd.Series) -> float:
        start = float(r["2021"])
        end = float(r["2024"])
        if start <= 0 or end <= 0:
            return 0.0
        return (end / start) ** (1 / 3) - 1

    return {
        "state_abbr": path.stem.split("_")[1],
        "real_gdp_2024": float(real_gdp["2024"]),
        "real_pce_2024": float(real_pce["2024"]),
        "employment_2024": float(employment["2024"]),
        "real_gdp_cagr_2021_2024": cagr(real_gdp),
        "real_pce_cagr_2021_2024": cagr(real_pce),
        "employment_cagr_2021_2024": cagr(employment),
    }


def build_inbound_touch_2031() -> pd.DataFrame:
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    interstate = routes.loc[(routes["year"] == 2031) & (routes["orig_state_fips"] != routes["dest_state_fips"])].copy()
    inbound = interstate.groupby("dest_state_fips", as_index=False)["prediction"].sum().rename(columns={"dest_state_fips": "state_fips", "prediction": "inbound_interstate_touch_2031"})
    return inbound


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [load_sas_metrics(path) for path in sorted(SAS_DIR.glob("SASUMMARY_*.csv"))]
    df = pd.DataFrame(rows)
    df = df.loc[df["state_abbr"].isin(STATE_ABBR_TO_FIPS)].copy()
    df = df.loc[~df["state_abbr"].isin(EXCLUDED)].copy()
    df["state_fips"] = df["state_abbr"].map(STATE_ABBR_TO_FIPS)
    df = df.merge(build_inbound_touch_2031(), on="state_fips", how="left")
    df["inbound_interstate_touch_2031"] = df["inbound_interstate_touch_2031"].fillna(0.0)
    df["demand_dependency"] = df["real_pce_2024"] / df["inbound_interstate_touch_2031"].replace({0: np.nan})
    df["demand_dependency"] = df["demand_dependency"].fillna(0.0)
    df["demand_dependency_winsor"] = winsorize(df["demand_dependency"])
    df["domestic_broad_demand_score"] = (
        1.25 * zscore(np.log1p(df["real_pce_2024"]))
        + 0.75 * zscore(np.log1p(df["real_gdp_2024"]))
        + 0.50 * zscore(np.log1p(df["employment_2024"]))
        + 1.00 * zscore(df["real_pce_cagr_2021_2024"])
        + 0.50 * zscore(df["real_gdp_cagr_2021_2024"])
        + 0.50 * zscore(df["employment_cagr_2021_2024"])
        + 1.00 * zscore(np.log1p(df["demand_dependency_winsor"]))
    )
    df = df.sort_values("domestic_broad_demand_score", ascending=False).reset_index(drop=True)
    df["rank_within_scenario"] = np.arange(1, len(df) + 1)
    df["severity"] = df["rank_within_scenario"].map(SEVERITY_BY_RANK)
    df.to_csv(OUT_DIR / "domestic_broad_demand_state_scores.csv", index=False)

    top = df.loc[df["rank_within_scenario"] <= 5].copy()
    top["scenario_name"] = "domestic_broad_demand_state_blocking"
    top["shock_family"] = "Demand shock"
    top["notes"] = "Broader domestic demand stress built from BEA state real PCE, GDP, employment, and recent growth."
    top.to_csv(OUT_DIR / "domestic_broad_demand_state_blocking_cases.csv", index=False)
    (OUT_DIR / "domestic_broad_demand_reference.md").write_text(
        "\n".join(
            [
                "# Domestic Broad Demand Shock Inputs",
                "",
                "Built from official BEA state annual summary data.",
                "- real PCE",
                "- real GDP",
                "- total employment",
                "- 2021-2024 growth of each series",
                "- demand dependency vs 2031 inbound interstate touch",
                "",
                "Top 5 selected states:",
            ]
            + [
                f"- {row.state_abbr}: severity={row.severity}, score={row.domestic_broad_demand_score:.3f}, real_pce={row.real_pce_2024:,.0f}, growth={row.real_pce_cagr_2021_2024:.2%}"
                for row in top.itertuples(index=False)
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

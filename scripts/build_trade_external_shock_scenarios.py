from __future__ import annotations

import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
EXPORT_DIR = PROJECT_ROOT / "external_data" / "Scenario_Tier1" / "Trade_External" / "Census_State_Exports"
BASE_ROUTE_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "trade_external_shock"

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
EXCLUDED_STATE_ABBRS = {"AK", "HI", "DC"}
CHINA_CODE = "5700"
FWF_SPECS = [(0, 4), (4, 8), (8, 10), (10, 14), (14, 16), (16, 31), (61, 76), (91, 106)]
FWF_NAMES = ["naics4", "cty_code", "state_abbr", "year", "month", "value_mo", "ves_val_mo", "cnt_val_mo"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def read_one_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        txt_name = next(name for name in zf.namelist() if name.lower().endswith(".txt"))
        with zf.open(txt_name) as fh:
            raw = fh.read()
    return pd.read_fwf(io.BytesIO(raw), colspecs=FWF_SPECS, names=FWF_NAMES, dtype={"cty_code": "string", "state_abbr": "string", "year": "int64", "month": "int64"})


def build_annual_trade_features() -> pd.DataFrame:
    frames = [read_one_zip(path) for path in sorted(EXPORT_DIR.glob("STNAICS_*.zip"))]
    trade = pd.concat(frames, ignore_index=True)
    trade["state_abbr"] = trade["state_abbr"].astype(str).str.strip()
    trade = trade.loc[trade["state_abbr"].isin(STATE_ABBR_TO_FIPS)].copy()
    for col in ["value_mo", "ves_val_mo", "cnt_val_mo"]:
        trade[col] = pd.to_numeric(trade[col], errors="coerce").fillna(0.0)
    annual = (
        trade.groupby(["state_abbr", "year"], as_index=False)
        .agg(
            total_export_value_usd=("value_mo", "sum"),
            vessel_export_value_usd=("ves_val_mo", "sum"),
            container_export_value_usd=("cnt_val_mo", "sum"),
            partner_count=("cty_code", "nunique"),
            naics4_count=("naics4", "nunique"),
        )
    )
    china = (
        trade.loc[trade["cty_code"].astype(str).str.zfill(4) == CHINA_CODE]
        .groupby(["state_abbr", "year"], as_index=False)["value_mo"]
        .sum()
        .rename(columns={"value_mo": "china_export_value_usd"})
    )
    annual = annual.merge(china, on=["state_abbr", "year"], how="left")
    annual["china_export_value_usd"] = annual["china_export_value_usd"].fillna(0.0)
    annual["china_share"] = annual["china_export_value_usd"] / annual["total_export_value_usd"].replace({0: np.nan})
    annual["vessel_share"] = annual["vessel_export_value_usd"] / annual["total_export_value_usd"].replace({0: np.nan})
    annual["container_share"] = annual["container_export_value_usd"] / annual["total_export_value_usd"].replace({0: np.nan})
    annual[["china_share", "vessel_share", "container_share"]] = annual[["china_share", "vessel_share", "container_share"]].fillna(0.0)
    annual["state_fips"] = annual["state_abbr"].map(STATE_ABBR_TO_FIPS)
    return annual


def build_state_interstate_touch_2031() -> pd.DataFrame:
    routes = pd.read_csv(BASE_ROUTE_PATH, dtype={"orig_state_fips": "string", "dest_state_fips": "string"})
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    interstate = routes.loc[(routes["year"] == 2031) & (routes["orig_state_fips"] != routes["dest_state_fips"])].copy()
    orig = interstate.groupby("orig_state_fips", as_index=False)["prediction"].sum().rename(columns={"orig_state_fips": "state_fips", "prediction": "orig_interstate_touch"})
    dest = interstate.groupby("dest_state_fips", as_index=False)["prediction"].sum().rename(columns={"dest_state_fips": "state_fips", "prediction": "dest_interstate_touch"})
    merged = orig.merge(dest, on="state_fips", how="outer").fillna(0.0)
    merged["total_interstate_touch_2031"] = merged["orig_interstate_touch"] + merged["dest_interstate_touch"]
    return merged[["state_fips", "total_interstate_touch_2031"]]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    annual = build_annual_trade_features()
    annual.to_csv(OUT_DIR / "trade_external_state_year_metrics.csv", index=False)

    latest = annual.loc[(annual["year"] == 2024) & (~annual["state_abbr"].isin(EXCLUDED_STATE_ABBRS))].copy()
    latest = latest.merge(build_state_interstate_touch_2031(), on="state_fips", how="left")
    latest["total_interstate_touch_2031"] = latest["total_interstate_touch_2031"].fillna(0.0)
    latest["export_dependency"] = latest["total_export_value_usd"] / latest["total_interstate_touch_2031"].replace({0: np.nan})
    latest["export_dependency"] = latest["export_dependency"].fillna(0.0)
    latest["export_dependency_winsor"] = winsorize(latest["export_dependency"])
    latest["trade_external_score"] = (
        1.50 * zscore(np.log1p(latest["total_export_value_usd"]))
        + 1.00 * zscore(latest["china_share"])
        + 0.75 * zscore(latest["vessel_share"])
        + 0.75 * zscore(latest["container_share"])
        + 1.00 * zscore(np.log1p(latest["export_dependency_winsor"]))
    )
    latest = latest.sort_values("trade_external_score", ascending=False).reset_index(drop=True)
    latest["rank_within_scenario"] = np.arange(1, len(latest) + 1)
    latest["severity"] = latest["rank_within_scenario"].map(SEVERITY_BY_RANK)
    latest.to_csv(OUT_DIR / "trade_external_state_scores.csv", index=False)

    top = latest.loc[latest["rank_within_scenario"] <= 5].copy()
    top["scenario_name"] = "trade_external_state_blocking"
    top["shock_family"] = "Trade/external network shock"
    top["notes"] = "First-pass trade exposure built from official Census state export files: total exports, China share, vessel share, container share, export dependency."
    top.to_csv(OUT_DIR / "trade_external_state_blocking_cases.csv", index=False)

    lines = [
        "# Trade / External Network Shock Inputs",
        "",
        "First-pass score uses official Census state export data only.",
        "- total export value",
        "- China export share",
        "- vessel export share",
        "- containerized vessel export share",
        "- export dependency against 2031 interstate touch volume",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.state_abbr} ({row.state_fips}): severity={row.severity}, score={row.trade_external_score:.3f}, "
            f"exports=${row.total_export_value_usd:,.0f}, china_share={row.china_share:.2%}, "
            f"vessel_share={row.vessel_share:.2%}, container_share={row.container_share:.2%}, "
            f"export_dependency={row.export_dependency:.4f}"
        )
    lines.extend(
        [
            "",
            "Limitations:",
            "- first pass does not yet include official state import panels",
            "- port congestion is proxied via vessel/container exposure rather than separate port delay data",
            "- same-state short-haul is intentionally excluded at simulation time",
        ]
    )
    (OUT_DIR / "trade_external_state_blocking_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

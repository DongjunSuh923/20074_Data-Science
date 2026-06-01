from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


WORK_ROOT = Path(__file__).resolve().parents[1]
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")

EXPORT_DIR = IDEA_ROOT / "external_data" / "Scenario_Tier1" / "Trade_External" / "Census_State_Exports"
IMPORT_DIR = WORK_ROOT / "external_data" / "Scenario_Tier2" / "Trade_External_Enhanced" / "Census_State_Imports"
BASE_ROUTE_PATH = IDEA_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "trade_external_enhanced_shock"

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
SEVERITY_BY_RANK = {1: "severe", 2: "medium", 3: "medium", 4: "mild", 5: "mild"}

CHINA_CODE = "5700"
CANADA_CODE = "1220"
MEXICO_CODE = "2010"
TRADE_WAR_WEIGHTS = {CHINA_CODE: 1.0, CANADA_CODE: 0.6, MEXICO_CODE: 0.6}

EXPORT_SPECS = [(0, 4), (4, 8), (8, 10), (10, 14), (14, 16), (16, 31), (61, 76), (91, 106)]
EXPORT_NAMES = ["naics4", "cty_code", "state_abbr", "year", "month", "value_mo", "ves_val_mo", "cnt_val_mo"]


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def winsorize(series: pd.Series, lower_q: float = 0.05, upper_q: float = 0.95) -> pd.Series:
    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    return series.clip(lower=lower, upper=upper)


def read_export_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        txt_name = next(name for name in zf.namelist() if name.lower().endswith(".txt"))
        raw = zf.read(txt_name)
    return pd.read_fwf(
        io.BytesIO(raw),
        colspecs=EXPORT_SPECS,
        names=EXPORT_NAMES,
        dtype={"cty_code": "string", "state_abbr": "string", "year": "int64", "month": "int64"},
    )


def parse_import_line(line: str) -> dict[str, object]:
    line = line.rstrip("\n")
    header = line[:16]
    fields = [line[16 + 15 * i : 16 + 15 * (i + 1)] for i in range(16)]
    nums = [int(f.strip() or "0") for f in fields]
    return {
        "naics4": header[:4],
        "cty_code": header[4:8],
        "state_abbr": header[8:10].strip(),
        "year": int(header[10:14]),
        "month": int(header[14:16]),
        "general_import_value_usd": nums[0],
        "general_import_shipping_wt": nums[1],
        "air_import_value_usd": nums[2],
        "air_import_shipping_wt": nums[3],
        "vessel_import_value_usd": nums[4],
        "vessel_import_shipping_wt": nums[5],
        "container_import_value_usd": nums[6],
        "container_import_shipping_wt": nums[7],
        "consumption_import_value_usd": nums[8],
        "consumption_import_shipping_wt": nums[9],
        "air_consumption_import_value_usd": nums[10],
        "air_consumption_import_shipping_wt": nums[11],
        "vessel_consumption_import_value_usd": nums[12],
        "vessel_consumption_import_shipping_wt": nums[13],
        "container_consumption_import_value_usd": nums[14],
        "container_consumption_import_shipping_wt": nums[15],
    }


def read_import_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        txt_name = next(name for name in zf.namelist() if name.lower().endswith(".txt"))
        with zf.open(txt_name) as fh:
            lines = fh.read().decode("utf-8", errors="ignore").splitlines()
    return pd.DataFrame(parse_import_line(line) for line in lines if line.strip())


def build_export_features() -> pd.DataFrame:
    frames = [read_export_zip(path) for path in sorted(EXPORT_DIR.glob("STNAICS_*.zip"))]
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
        )
    )
    for code, label in [("5700", "china"), ("1220", "canada"), ("2010", "mexico")]:
        sub = (
            trade.loc[trade["cty_code"].astype(str).str.zfill(4) == code]
            .groupby(["state_abbr", "year"], as_index=False)["value_mo"]
            .sum()
            .rename(columns={"value_mo": f"{label}_export_value_usd"})
        )
        annual = annual.merge(sub, on=["state_abbr", "year"], how="left")
    return annual.fillna(0.0)


def build_import_features() -> pd.DataFrame:
    frames = [read_import_zip(path) for path in sorted(IMPORT_DIR.glob("ISNAICS_*.zip"))]
    trade = pd.concat(frames, ignore_index=True)
    trade["state_abbr"] = trade["state_abbr"].astype(str).str.strip()
    trade = trade.loc[trade["state_abbr"].isin(STATE_ABBR_TO_FIPS)].copy()
    annual = (
        trade.groupby(["state_abbr", "year"], as_index=False)
        .agg(
            total_import_value_usd=("general_import_value_usd", "sum"),
            vessel_import_value_usd=("vessel_import_value_usd", "sum"),
            container_import_value_usd=("container_import_value_usd", "sum"),
        )
    )
    for code, label in [("5700", "china"), ("1220", "canada"), ("2010", "mexico")]:
        sub = (
            trade.loc[trade["cty_code"].astype(str).str.zfill(4) == code]
            .groupby(["state_abbr", "year"], as_index=False)["general_import_value_usd"]
            .sum()
            .rename(columns={"general_import_value_usd": f"{label}_import_value_usd"})
        )
        annual = annual.merge(sub, on=["state_abbr", "year"], how="left")
    return annual.fillna(0.0)


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
    merged["inbound_interstate_touch_2031"] = merged["dest_interstate_touch"]
    merged["outbound_interstate_touch_2031"] = merged["orig_interstate_touch"]
    return merged


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    exports = build_export_features()
    imports = build_import_features()
    annual = exports.merge(imports, on=["state_abbr", "year"], how="outer").fillna(0.0)

    for label in ["china", "canada", "mexico"]:
        annual[f"{label}_trade_value_usd"] = annual[f"{label}_export_value_usd"] + annual[f"{label}_import_value_usd"]
    annual["total_trade_value_usd"] = annual["total_export_value_usd"] + annual["total_import_value_usd"]
    annual["total_vessel_trade_value_usd"] = annual["vessel_export_value_usd"] + annual["vessel_import_value_usd"]
    annual["total_container_trade_value_usd"] = annual["container_export_value_usd"] + annual["container_import_value_usd"]
    annual["import_intensity"] = annual["total_import_value_usd"] / annual["total_trade_value_usd"].replace({0: np.nan})
    annual["export_intensity"] = annual["total_export_value_usd"] / annual["total_trade_value_usd"].replace({0: np.nan})
    annual["maritime_share"] = annual["total_vessel_trade_value_usd"] / annual["total_trade_value_usd"].replace({0: np.nan})
    annual["container_share"] = annual["total_container_trade_value_usd"] / annual["total_trade_value_usd"].replace({0: np.nan})
    for label in ["china", "canada", "mexico"]:
        annual[f"{label}_trade_share"] = annual[f"{label}_trade_value_usd"] / annual["total_trade_value_usd"].replace({0: np.nan})
    annual = annual.fillna(0.0)
    annual["trade_war_partner_exposure"] = (
        TRADE_WAR_WEIGHTS[CHINA_CODE] * annual["china_trade_share"]
        + TRADE_WAR_WEIGHTS[CANADA_CODE] * annual["canada_trade_share"]
        + TRADE_WAR_WEIGHTS[MEXICO_CODE] * annual["mexico_trade_share"]
    )
    annual["state_fips"] = annual["state_abbr"].map(STATE_ABBR_TO_FIPS)
    annual.to_csv(OUT_DIR / "trade_external_enhanced_state_year_metrics.csv", index=False)

    latest = annual.loc[(annual["year"] == 2024) & (~annual["state_abbr"].isin(EXCLUDED_STATE_ABBRS))].copy()
    latest = latest.merge(build_state_interstate_touch_2031(), on="state_fips", how="left")
    latest[["total_interstate_touch_2031", "inbound_interstate_touch_2031", "outbound_interstate_touch_2031"]] = latest[
        ["total_interstate_touch_2031", "inbound_interstate_touch_2031", "outbound_interstate_touch_2031"]
    ].fillna(0.0)
    latest["trade_dependency"] = latest["total_trade_value_usd"] / latest["total_interstate_touch_2031"].replace({0: np.nan})
    latest["import_dependency"] = latest["total_import_value_usd"] / latest["inbound_interstate_touch_2031"].replace({0: np.nan})
    latest["export_dependency"] = latest["total_export_value_usd"] / latest["outbound_interstate_touch_2031"].replace({0: np.nan})
    for col in ["trade_dependency", "import_dependency", "export_dependency"]:
        latest[col] = latest[col].fillna(0.0)
        latest[f"{col}_winsor"] = winsorize(latest[col])

    latest["trade_external_enhanced_score"] = (
        1.25 * zscore(np.log1p(latest["total_trade_value_usd"]))
        + 1.00 * zscore(latest["trade_war_partner_exposure"])
        + 0.75 * zscore(latest["maritime_share"])
        + 0.75 * zscore(latest["container_share"])
        + 0.75 * zscore(latest["import_intensity"])
        + 1.00 * zscore(np.log1p(latest["trade_dependency_winsor"]))
        + 0.50 * zscore(np.log1p(latest["import_dependency_winsor"]))
    )
    latest = latest.sort_values("trade_external_enhanced_score", ascending=False).reset_index(drop=True)
    latest["rank_within_scenario"] = np.arange(1, len(latest) + 1)
    latest["severity"] = latest["rank_within_scenario"].map(SEVERITY_BY_RANK)
    latest.to_csv(OUT_DIR / "trade_external_enhanced_state_scores.csv", index=False)

    top = latest.loc[latest["rank_within_scenario"] <= 5].copy()
    top["scenario_name"] = "trade_external_enhanced_state_blocking"
    top["shock_family"] = "Trade/external network shock"
    top["notes"] = (
        "Enhanced trade exposure built from official Census state export/import files plus USTR 2018-2019 "
        "China Section 301 and Canada/Mexico Section 232 trade-war overlay weights."
    )
    top.to_csv(OUT_DIR / "trade_external_enhanced_state_blocking_cases.csv", index=False)

    lines = [
        "# Enhanced Trade / External Network Shock Inputs",
        "",
        "Enhanced score uses official Census state export + import data and trade-war partner exposure.",
        "- total trade value (exports + imports)",
        "- China / Canada / Mexico combined trade exposure",
        "- vessel trade share",
        "- containerized trade share",
        "- import intensity",
        "- trade dependency against 2031 interstate touch volume",
        "",
        "Trade-war overlay sources:",
        "- USTR China Section 301 tariff actions (2018-2020)",
        "- USTR Canada/Mexico Section 232 steel & aluminum tariff episode (2018-2019)",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.state_abbr} ({row.state_fips}): severity={row.severity}, score={row.trade_external_enhanced_score:.3f}, "
            f"trade=${row.total_trade_value_usd:,.0f}, import_share={row.import_intensity:.2%}, "
            f"china={row.china_trade_share:.2%}, canada={row.canada_trade_share:.2%}, mexico={row.mexico_trade_share:.2%}, "
            f"maritime={row.maritime_share:.2%}, container={row.container_share:.2%}"
        )
    (OUT_DIR / "trade_external_enhanced_state_blocking_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

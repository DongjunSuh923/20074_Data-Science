from __future__ import annotations

import gzip
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

DATA_ROOT = PROJECT_ROOT / "external_data" / "Scenario_Tier1" / "Natural_Disaster"
STORM_DIR = DATA_ROOT / "NOAA_StormEvents"
AQI_DIR = DATA_ROOT / "EPA_AQI"
CLIMATE_DIR = DATA_ROOT / "NOAA_ClimateAtAGlance"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity"

HAZARD_MAP = {
    "snow_heat": {
        "storm_types": {"Winter Storm", "Heavy Snow", "Blizzard", "Ice Storm", "Heat", "Excessive Heat", "Drought"},
        "aqi_weight": 0.0,
    },
    "tornado": {
        "storm_types": {"Tornado"},
        "aqi_weight": 0.0,
    },
    "wildfire_smoke": {
        "storm_types": {"Wildfire"},
        "aqi_weight": 1.0,
    },
}
STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08",
    "CT": "09", "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15",
    "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
    "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}
STATE_NAME_TO_FIPS = {
    "ALABAMA": "01", "ALASKA": "02", "ARIZONA": "04", "ARKANSAS": "05", "CALIFORNIA": "06",
    "COLORADO": "08", "CONNECTICUT": "09", "DELAWARE": "10", "DISTRICT OF COLUMBIA": "11",
    "FLORIDA": "12", "GEORGIA": "13", "HAWAII": "15", "IDAHO": "16", "ILLINOIS": "17",
    "INDIANA": "18", "IOWA": "19", "KANSAS": "20", "KENTUCKY": "21", "LOUISIANA": "22",
    "MAINE": "23", "MARYLAND": "24", "MASSACHUSETTS": "25", "MICHIGAN": "26", "MINNESOTA": "27",
    "MISSISSIPPI": "28", "MISSOURI": "29", "MONTANA": "30", "NEBRASKA": "31", "NEVADA": "32",
    "NEW HAMPSHIRE": "33", "NEW JERSEY": "34", "NEW MEXICO": "35", "NEW YORK": "36",
    "NORTH CAROLINA": "37", "NORTH DAKOTA": "38", "OHIO": "39", "OKLAHOMA": "40", "OREGON": "41",
    "PENNSYLVANIA": "42", "RHODE ISLAND": "44", "SOUTH CAROLINA": "45", "SOUTH DAKOTA": "46",
    "TENNESSEE": "47", "TEXAS": "48", "UTAH": "49", "VERMONT": "50", "VIRGINIA": "51",
    "WASHINGTON": "53", "WEST VIRGINIA": "54", "WISCONSIN": "55", "WYOMING": "56",
}
SEVERITY_BY_RANK = {
    1: "severe",
    2: "medium",
    3: "medium",
    4: "mild",
    5: "mild",
}


def parse_damage(value: str) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value).strip().upper()
    if not text or text in {"0", "0.00K", "0.00M", "0.00B"}:
        return 0.0
    mult = 1.0
    if text.endswith("K"):
        mult = 1_000.0
        text = text[:-1]
    elif text.endswith("M"):
        mult = 1_000_000.0
        text = text[:-1]
    elif text.endswith("B"):
        mult = 1_000_000_000.0
        text = text[:-1]
    try:
        return float(text) * mult
    except ValueError:
        return 0.0


def load_storm_events() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    keep_cols = ["STATE", "YEAR", "EVENT_TYPE", "DAMAGE_PROPERTY", "DAMAGE_CROPS"]
    for path in sorted(STORM_DIR.glob("stormevents_details_*.csv.gz")):
        with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore") as fh:
            df = pd.read_csv(fh, usecols=keep_cols, low_memory=False)
        df["state_key"] = df["STATE"].astype(str).str.upper().str.strip()
        df["state_fips"] = (
            df["state_key"].map(STATE_NAME_TO_FIPS)
            .fillna(df["state_key"].map(STATE_ABBR_TO_FIPS))
        )
        df["year"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df["property_damage"] = df["DAMAGE_PROPERTY"].map(parse_damage)
        df["crop_damage"] = df["DAMAGE_CROPS"].map(parse_damage)
        df["total_damage"] = df["property_damage"] + df["crop_damage"]
        frames.append(df[["state_fips", "year", "EVENT_TYPE", "total_damage"]])
    return pd.concat(frames, ignore_index=True)


def load_aqi_pm25_days() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(AQI_DIR.glob("annual_aqi_by_county_*.zip")):
        year = int(path.stem.rsplit("_", 1)[-1])
        with zipfile.ZipFile(path) as zf:
            member = zf.namelist()[0]
            with zf.open(member) as fh:
                df = pd.read_csv(fh, low_memory=False)
        cols = {c.lower(): c for c in df.columns}
        state_col = cols.get("state code")
        pm25_col = cols.get("days pm2.5")
        if state_col is None or pm25_col is None:
            continue
        keep = df[[state_col, pm25_col]].copy()
        keep.columns = ["state_fips_num", "days_pm25"]
        keep["state_fips"] = keep["state_fips_num"].astype(str).str.zfill(2)
        keep["year"] = year
        keep["days_pm25"] = pd.to_numeric(keep["days_pm25"], errors="coerce").fillna(0.0)
        frames.append(keep[["state_fips", "year", "days_pm25"]])
    if not frames:
        return pd.DataFrame(columns=["state_fips", "year", "days_pm25"])
    state_year = (
        pd.concat(frames, ignore_index=True)
        .groupby(["state_fips", "year"], as_index=False)["days_pm25"]
        .sum()
    )
    return state_year


def load_climate() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(CLIMATE_DIR.glob("*_average_temperature_2018_2025.csv")):
        state_abbr = path.name.split("_")[0]
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Date" not in df.columns or "Value" not in df.columns:
            continue
        part = df[["Date", "Value"]].copy()
        part["year"] = pd.to_numeric(part["Date"], errors="coerce").astype("Int64")
        part["temperature_value"] = pd.to_numeric(part["Value"], errors="coerce")
        part["state_fips"] = STATE_ABBR_TO_FIPS.get(state_abbr)
        frames.append(part[["state_fips", "year", "temperature_value"]])
    if not frames:
        return pd.DataFrame(columns=["state_fips", "year", "temperature_value"])
    return pd.concat(frames, ignore_index=True)


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def build_hazard_scores(storms: pd.DataFrame, aqi: pd.DataFrame, climate: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    summary_rows = []
    for hazard_name, cfg in HAZARD_MAP.items():
        subset = storms.loc[storms["EVENT_TYPE"].isin(cfg["storm_types"])].copy()
        grouped = subset.groupby(["state_fips", "year"], as_index=False).agg(
            event_count=("EVENT_TYPE", "size"),
            total_damage=("total_damage", "sum"),
        )
        merged = grouped.merge(aqi, on=["state_fips", "year"], how="left")
        merged = merged.merge(climate, on=["state_fips", "year"], how="left")
        merged["days_pm25"] = merged["days_pm25"].fillna(0.0)
        merged["temperature_value"] = merged["temperature_value"].fillna(0.0)
        state_summary = (
            merged.groupby("state_fips", as_index=False)
            .agg(
                avg_event_count=("event_count", "mean"),
                avg_total_damage=("total_damage", "mean"),
                avg_days_pm25=("days_pm25", "mean"),
                avg_temperature=("temperature_value", "mean"),
            )
        )
        state_summary["hazard"] = hazard_name
        state_summary["score"] = (
            zscore(state_summary["avg_event_count"].fillna(0.0))
            + zscore(np.log1p(state_summary["avg_total_damage"].fillna(0.0)))
            + cfg["aqi_weight"] * zscore(state_summary["avg_days_pm25"].fillna(0.0))
        )
        state_summary = state_summary.sort_values("score", ascending=False).reset_index(drop=True)
        state_summary["rank_within_hazard"] = np.arange(1, len(state_summary) + 1)
        state_summary["severity"] = state_summary["rank_within_hazard"].map(SEVERITY_BY_RANK)
        summary_rows.append(state_summary)
        all_rows.append(merged.assign(hazard=hazard_name))
    return pd.concat(all_rows, ignore_index=True), pd.concat(summary_rows, ignore_index=True)


def build_scenario_cases(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hazard in summary["hazard"].unique():
        top = summary.loc[(summary["hazard"] == hazard) & (summary["rank_within_hazard"] <= 5)].copy()
        top = top.loc[top["severity"].notna()]
        for row in top.itertuples(index=False):
            rows.append(
                {
                    "scenario_name": f"{hazard}_state_blocking",
                    "shock_family": "Capacity shock",
                    "hazard": hazard,
                    "state_fips": row.state_fips,
                    "severity": row.severity,
                    "historical_score": row.score,
                    "avg_event_count": row.avg_event_count,
                    "avg_total_damage": row.avg_total_damage,
                    "avg_days_pm25": row.avg_days_pm25,
                }
            )
    return pd.DataFrame(rows)


def write_markdown(summary: pd.DataFrame, cases: pd.DataFrame) -> None:
    lines = ["# Natural Disaster Capacity Shock Inputs", ""]
    for hazard in ["snow_heat", "tornado", "wildfire_smoke"]:
        lines.append(f"## {hazard}")
        top = summary.loc[summary["hazard"] == hazard].head(5)
        for row in top.itertuples(index=False):
            sev = row.severity if pd.notna(row.severity) else "none"
            lines.append(
                f"- {row.state_fips}: score={row.score:.3f}, avg_events={row.avg_event_count:.2f}, "
                f"avg_damage=${row.avg_total_damage:,.0f}, avg_pm25_days={row.avg_days_pm25:.2f}, severity={sev}"
            )
        lines.append("")
    lines.append("## Selected Blocking Cases")
    for row in cases.itertuples(index=False):
        lines.append(f"- {row.scenario_name}: {row.state_fips} -> {row.severity}")
    (OUT_DIR / "natural_disaster_capacity_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    storms = load_storm_events()
    aqi = load_aqi_pm25_days()
    climate = load_climate()
    detailed, summary = build_hazard_scores(storms, aqi, climate)
    cases = build_scenario_cases(summary)
    detailed.to_csv(OUT_DIR / "natural_disaster_state_year_metrics.csv", index=False)
    summary.to_csv(OUT_DIR / "natural_disaster_state_scores.csv", index=False)
    cases.to_csv(OUT_DIR / "natural_disaster_state_blocking_cases.csv", index=False)
    write_markdown(summary, cases)


if __name__ == "__main__":
    main()

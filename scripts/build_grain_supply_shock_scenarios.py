from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
USDA_CROP_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "energy_agri_feature_prep" / "usda_state_crop_features_2018_2024.csv"
USDA_EXT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "gravel_cereal_extended_feature_prep" / "state_usda_cereal_extended_features_2018_2024.csv"
STORM_DIR = PROJECT_ROOT / "external_data" / "Scenario_Tier1" / "Natural_Disaster" / "NOAA_StormEvents"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "grain_supply_shock"

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "DISTRICT OF COLUMBIA": "DC",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL",
    "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA",
    "MAINE": "ME", "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK", "OREGON": "OR",
    "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD",
    "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA",
    "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}
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
SEVERITY_BY_RANK = {1: "severe", 2: "medium", 3: "medium", 4: "mild", 5: "mild"}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def load_drought_counts() -> pd.DataFrame:
    frames = []
    for path in sorted(STORM_DIR.glob("stormevents_details_*.csv.gz")):
        with gzip.open(path, mode="rt", encoding="utf-8", errors="ignore") as fh:
            df = pd.read_csv(fh, usecols=["STATE", "YEAR", "EVENT_TYPE"], low_memory=False)
        df["state_abbr"] = df["STATE"].astype(str).str.upper().str.strip().map(STATE_NAME_TO_ABBR)
        df["year"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        drought = df.loc[df["EVENT_TYPE"] == "Drought", ["state_abbr", "year"]].copy()
        drought["drought_events"] = 1
        frames.append(drought)
    all_drought = pd.concat(frames, ignore_index=True)
    return all_drought.groupby(["state_abbr", "year"], as_index=False)["drought_events"].sum()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    crop = pd.read_csv(USDA_CROP_PATH)
    ext = pd.read_csv(USDA_EXT_PATH)
    drought = load_drought_counts()

    crop_2024 = crop.loc[crop["year"] == 2024, ["state_abbr", "grain_production_index_bu", "grain_production_index_bu_growth"]].copy()
    ext_2024 = ext.loc[ext["year"] == 2024, ["state_abbr", "grain_harvested_area_index", "grain_planted_area_index"]].copy()
    drought_recent = drought.loc[drought["year"].between(2022, 2025)].groupby("state_abbr", as_index=False)["drought_events"].mean()
    drought_recent = drought_recent.rename(columns={"drought_events": "avg_drought_events_2022_2025"})

    merged = crop_2024.merge(ext_2024, on="state_abbr", how="left").merge(drought_recent, on="state_abbr", how="left")
    merged["avg_drought_events_2022_2025"] = merged["avg_drought_events_2022_2025"].fillna(0.0)
    merged["grain_production_index_bu"] = pd.to_numeric(merged["grain_production_index_bu"], errors="coerce").fillna(0.0)
    merged["grain_harvested_area_index"] = pd.to_numeric(merged["grain_harvested_area_index"], errors="coerce").fillna(0.0)
    merged["grain_production_index_bu_growth"] = pd.to_numeric(merged["grain_production_index_bu_growth"], errors="coerce").fillna(0.0)

    merged["grain_supply_score"] = (
        1.5 * zscore(np.log1p(merged["grain_production_index_bu"]))
        + 1.0 * zscore(np.log1p(merged["grain_harvested_area_index"]))
        + 0.5 * zscore(merged["avg_drought_events_2022_2025"])
        + 0.25 * zscore(-merged["grain_production_index_bu_growth"])
    )
    merged = merged.sort_values("grain_supply_score", ascending=False).reset_index(drop=True)
    merged["rank_within_scenario"] = np.arange(1, len(merged) + 1)
    merged["severity"] = merged["rank_within_scenario"].map(SEVERITY_BY_RANK)
    merged["state_fips"] = merged["state_abbr"].map(STATE_ABBR_TO_FIPS)

    cases = merged.loc[merged["rank_within_scenario"] <= 5, ["state_fips", "state_abbr", "severity", "grain_supply_score", "grain_production_index_bu", "grain_harvested_area_index", "avg_drought_events_2022_2025"]].copy()
    cases["scenario_name"] = "grain_supply_state_blocking"
    cases["shock_family"] = "Commodity-specific supply shock"
    cases["commodity_filter"] = "2"
    cases = cases[["scenario_name", "shock_family", "commodity_filter", "state_fips", "state_abbr", "severity", "grain_supply_score", "grain_production_index_bu", "grain_harvested_area_index", "avg_drought_events_2022_2025"]]

    merged.to_csv(OUT_DIR / "grain_supply_state_scores.csv", index=False)
    cases.to_csv(OUT_DIR / "grain_supply_state_blocking_cases.csv", index=False)

    lines = ["# Grain Supply Shock Inputs", ""]
    for row in cases.itertuples(index=False):
        lines.append(
            f"- {row.state_abbr}: severity={row.severity}, score={row.grain_supply_score:.3f}, "
            f"grain_production={row.grain_production_index_bu:,.0f}, harvested_area={row.grain_harvested_area_index:,.0f}, "
            f"avg_drought_events={row.avg_drought_events_2022_2025:.2f}"
        )
    (OUT_DIR / "grain_supply_state_blocking_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

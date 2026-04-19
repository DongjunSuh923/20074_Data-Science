from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "faf5_long_final.csv"
STATE_EXTERNAL_PATH = PROJECT_ROOT / "outputs" / "external_features" / "state_external_features.csv"
ZONE_LOOKUP_PATH = PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "lookups" / "domestic_zone_lookup.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model"

OUTPUT_DATASET = OUTPUT_DIR / "scenario_dataset_state_to_state.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "scenario_dataset_summary.json"

GROUP_COLS = [
    "orig_state_fips",
    "dest_state_fips",
    "sctg2",
    "dist_band",
    "trade_type",
    "year",
    "year_index",
    "split",
]

SUM_COLS = ["target_tons"]

STATE_FEATURE_COLS = [
    "gdp_total",
    "gdp_manufacturing",
    "gdp_wholesale",
    "gdp_transport_warehousing",
    "gdp_total_growth",
    "gdp_manufacturing_growth",
    "gdp_wholesale_growth",
    "gdp_transport_warehousing_growth",
    "gdp_mfg_share",
    "gdp_wholesale_share",
    "gdp_transport_warehousing_share",
    "cbp_total_emp",
    "cbp_mfg_emp",
    "cbp_wholesale_emp",
    "cbp_transport_emp",
    "cbp_warehousing_emp",
    "cbp_total_est",
    "cbp_warehousing_est",
]


def assign_split(year: int) -> str:
    if year <= 2021:
        return "train"
    if year == 2022:
        return "validation"
    if year == 2023:
        return "test_2023"
    if year == 2024:
        return "test_2024"
    return "other"


def load_base_long() -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_PATH,
        usecols=["dms_orig", "dms_dest", "sctg2", "trade_type", "dist_band", "year", "tons"],
        dtype={
            "dms_orig": "string",
            "dms_dest": "string",
            "sctg2": "string",
            "trade_type": "string",
            "dist_band": "string",
        },
        low_memory=False,
    )
    df = df.rename(columns={"tons": "target_tons"})
    df["dms_orig"] = df["dms_orig"].str.zfill(3)
    df["dms_dest"] = df["dms_dest"].str.zfill(3)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.loc[df["year"].between(2018, 2024, inclusive="both")].copy()
    df["year"] = df["year"].astype(int)
    df["year_index"] = df["year"] - 2018
    df["split"] = df["year"].map(assign_split)
    df = df.loc[df["split"] != "other"].copy()
    return df


def attach_states(df: pd.DataFrame) -> pd.DataFrame:
    zone_lookup = pd.read_csv(
        ZONE_LOOKUP_PATH,
        usecols=["dms_code", "state_fips", "state_name"],
        dtype={"dms_code": "string", "state_fips": "string"},
    )
    zone_lookup["dms_code"] = zone_lookup["dms_code"].str.zfill(3)
    zone_lookup["state_fips"] = zone_lookup["state_fips"].str.zfill(2)

    orig = zone_lookup.rename(
        columns={
            "dms_code": "dms_orig",
            "state_fips": "orig_state_fips",
            "state_name": "orig_state_name",
        }
    )
    dest = zone_lookup.rename(
        columns={
            "dms_code": "dms_dest",
            "state_fips": "dest_state_fips",
            "state_name": "dest_state_name",
        }
    )

    merged = df.merge(orig, on="dms_orig", how="left")
    merged = merged.merge(dest, on="dms_dest", how="left")
    return merged


def load_state_features() -> pd.DataFrame:
    state_features = pd.read_csv(
        STATE_EXTERNAL_PATH,
        usecols=["state_fips", "year", "state_name"] + STATE_FEATURE_COLS,
        dtype={"state_fips": "string"},
        low_memory=False,
    )
    state_features["state_fips"] = state_features["state_fips"].str.zfill(2)
    return state_features


def attach_state_features(df: pd.DataFrame, state_features: pd.DataFrame) -> pd.DataFrame:
    orig = state_features.add_prefix("orig_").rename(
        columns={"orig_state_fips": "orig_state_fips", "orig_year": "year"}
    )
    dest = state_features.add_prefix("dest_").rename(
        columns={"dest_state_fips": "dest_state_fips", "dest_year": "year"}
    )
    merged = df.merge(orig, on=["orig_state_fips", "year"], how="left")
    merged = merged.merge(dest, on=["dest_state_fips", "year"], how="left")

    merged["route_gdp_total_sum"] = merged["orig_gdp_total"] + merged["dest_gdp_total"]
    merged["route_gdp_total_gap"] = (merged["orig_gdp_total"] - merged["dest_gdp_total"]).abs()
    merged["route_cbp_total_emp_sum"] = merged["orig_cbp_total_emp"] + merged["dest_cbp_total_emp"]
    merged["route_cbp_warehousing_emp_sum"] = merged["orig_cbp_warehousing_emp"] + merged["dest_cbp_warehousing_emp"]
    merged["route_cbp_transport_emp_sum"] = merged["orig_cbp_transport_emp"] + merged["dest_cbp_transport_emp"]
    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = load_base_long()
    with_states = attach_states(base)
    state_features = load_state_features()
    enriched = attach_state_features(with_states, state_features)

    agg_map = {col: "sum" for col in SUM_COLS}
    first_cols = [
        "orig_gdp_total",
        "dest_gdp_total",
        "orig_gdp_total_growth",
        "dest_gdp_total_growth",
        "orig_gdp_manufacturing",
        "dest_gdp_manufacturing",
        "orig_gdp_wholesale",
        "dest_gdp_wholesale",
        "orig_gdp_transport_warehousing",
        "dest_gdp_transport_warehousing",
        "orig_gdp_mfg_share",
        "dest_gdp_mfg_share",
        "orig_gdp_wholesale_share",
        "dest_gdp_wholesale_share",
        "orig_gdp_transport_warehousing_share",
        "dest_gdp_transport_warehousing_share",
        "orig_cbp_total_emp",
        "dest_cbp_total_emp",
        "orig_cbp_mfg_emp",
        "dest_cbp_mfg_emp",
        "orig_cbp_wholesale_emp",
        "dest_cbp_wholesale_emp",
        "orig_cbp_transport_emp",
        "dest_cbp_transport_emp",
        "orig_cbp_warehousing_emp",
        "dest_cbp_warehousing_emp",
        "orig_cbp_total_est",
        "dest_cbp_total_est",
        "orig_cbp_warehousing_est",
        "dest_cbp_warehousing_est",
        "route_gdp_total_sum",
        "route_gdp_total_gap",
        "route_cbp_total_emp_sum",
        "route_cbp_warehousing_emp_sum",
        "route_cbp_transport_emp_sum",
    ]
    agg_map.update({col: "first" for col in first_cols})

    scenario_df = (
        enriched.groupby(GROUP_COLS, dropna=False, as_index=False)
        .agg(agg_map)
        .sort_values(GROUP_COLS)
        .reset_index(drop=True)
    )
    scenario_df["target_log_tons"] = np.log1p(scenario_df["target_tons"])
    scenario_df["state_pair"] = scenario_df["orig_state_fips"] + "-" + scenario_df["dest_state_fips"]

    scenario_df.to_csv(OUTPUT_DATASET, index=False)

    summary = {
        "dataset_path": str(OUTPUT_DATASET),
        "row_count": int(scenario_df.shape[0]),
        "years": sorted(scenario_df["year"].unique().tolist()),
        "state_pairs": int(scenario_df["state_pair"].nunique()),
        "commodities": int(scenario_df["sctg2"].nunique()),
        "splits": scenario_df["split"].value_counts().to_dict(),
    }
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import site
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_SITE = site.getusersitepackages()
if USER_SITE and USER_SITE not in sys.path:
    sys.path.append(USER_SITE)

import pandas as pd
import xlrd


EXTERNAL_DIR = PROJECT_ROOT / "external_data"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
LOOKUP_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "lookups"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_features"

MODEL_READY_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"
FORECAST_2025_PATH = FEATURE_DIR / "feature_dataset_forecast_2025.csv"

STATE_TO_PADD = {
    "CT": "PADD1A", "ME": "PADD1A", "MA": "PADD1A", "NH": "PADD1A", "RI": "PADD1A", "VT": "PADD1A",
    "DE": "PADD1B", "DC": "PADD1B", "MD": "PADD1B", "NJ": "PADD1B", "NY": "PADD1B", "PA": "PADD1B",
    "FL": "PADD1C", "GA": "PADD1C", "NC": "PADD1C", "SC": "PADD1C", "VA": "PADD1C", "WV": "PADD1C",
    "IL": "PADD2", "IN": "PADD2", "IA": "PADD2", "KS": "PADD2", "KY": "PADD2", "MI": "PADD2", "MN": "PADD2",
    "MO": "PADD2", "NE": "PADD2", "ND": "PADD2", "OH": "PADD2", "OK": "PADD2", "SD": "PADD2", "TN": "PADD2", "WI": "PADD2",
    "AL": "PADD3", "AR": "PADD3", "LA": "PADD3", "MS": "PADD3", "NM": "PADD3", "TX": "PADD3",
    "CO": "PADD4", "ID": "PADD4", "MT": "PADD4", "UT": "PADD4", "WY": "PADD4",
    "AK": "PADD5X", "AZ": "PADD5X", "HI": "PADD5X", "NV": "PADD5X", "OR": "PADD5X", "WA": "PADD5X",
    "CA": "CA",
}

PADD_COLUMN_MAP = {
    "US": 1,
    "PADD1": 2,
    "PADD1A": 3,
    "PADD1B": 4,
    "PADD1C": 5,
    "PADD2": 6,
    "PADD3": 7,
    "PADD4": 8,
    "PADD5": 9,
    "CA": 10,
    "PADD5X": 11,
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


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def excel_date_to_datetime(serial: float) -> datetime:
    return datetime(1899, 12, 30) + timedelta(days=float(serial))


def load_state_lookup() -> pd.DataFrame:
    state_lookup = pd.read_csv(LOOKUP_DIR / "state_lookup.csv", dtype={"state_fips": str})
    state_lookup["state_fips"] = state_lookup["state_fips"].astype(str).str.zfill(2)
    state_lookup["state_abbr"] = state_lookup["state_fips"].map(STATE_ABBR_MAP)
    state_lookup["padd_region"] = state_lookup["state_abbr"].map(STATE_TO_PADD)
    return state_lookup


def parse_diesel_monthly() -> pd.DataFrame:
    diesel_path = next(EXTERNAL_DIR.glob("psw18vwall*.xls"))
    book = xlrd.open_workbook(str(diesel_path))
    sh = book.sheet_by_name("Data 6")
    records = []
    for r in range(3, sh.nrows):
        row = sh.row_values(r)
        if not row or row[0] == "":
            continue
        dt = excel_date_to_datetime(row[0])
        if not (2018 <= dt.year <= 2026):
            continue
        for region, col_idx in PADD_COLUMN_MAP.items():
            if col_idx >= len(row) or row[col_idx] == "":
                continue
            records.append(
                {
                    "year": dt.year,
                    "month": dt.month,
                    "fuel_region": region,
                    "diesel_price": float(row[col_idx]),
                }
            )
    monthly = pd.DataFrame(records)
    monthly.to_csv(OUTPUT_DIR / "diesel_monthly_by_region.csv", index=False)
    return monthly


def build_diesel_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    annual = (
        monthly.groupby(["fuel_region", "year"], as_index=False)["diesel_price"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "mean": "diesel_price_mean",
                "std": "diesel_price_std",
                "min": "diesel_price_min",
                "max": "diesel_price_max",
            }
        )
    )
    annual["diesel_price_yoy_growth"] = annual.groupby("fuel_region")["diesel_price_mean"].pct_change()
    annual.to_csv(OUTPUT_DIR / "diesel_annual_by_region.csv", index=False)
    return annual


def parse_wti_annual() -> pd.DataFrame:
    wti_path = next(EXTERNAL_DIR.glob("RWTCa*.xls"))
    book = xlrd.open_workbook(str(wti_path))
    sh = book.sheet_by_name("Data 1")
    rows = []
    for r in range(3, sh.nrows):
        row = sh.row_values(r)
        if not row or row[0] == "":
            continue
        dt = excel_date_to_datetime(row[0])
        if 2018 <= dt.year <= 2025:
            rows.append({"year": dt.year, "wti_price_annual": float(row[1])})
    wti = pd.DataFrame(rows).sort_values("year")
    wti["wti_price_yoy_growth"] = wti["wti_price_annual"].pct_change()
    wti.to_csv(OUTPUT_DIR / "wti_annual.csv", index=False)
    return wti


def merge_fuel(df: pd.DataFrame, annual: pd.DataFrame, wti: pd.DataFrame, state_lookup: pd.DataFrame) -> pd.DataFrame:
    orig_lookup = state_lookup[["state_fips", "state_abbr", "padd_region"]].rename(
        columns={"state_fips": "orig_state_fips", "state_abbr": "orig_state_abbr", "padd_region": "orig_padd_region"}
    )
    dest_lookup = state_lookup[["state_fips", "state_abbr", "padd_region"]].rename(
        columns={"state_fips": "dest_state_fips", "state_abbr": "dest_state_abbr", "padd_region": "dest_padd_region"}
    )
    merged = df.merge(orig_lookup, on="orig_state_fips", how="left").merge(dest_lookup, on="dest_state_fips", how="left")

    orig_annual = annual.rename(
        columns={
            "fuel_region": "orig_padd_region",
            "diesel_price_mean": "orig_diesel_price_mean",
            "diesel_price_std": "orig_diesel_price_std",
            "diesel_price_min": "orig_diesel_price_min",
            "diesel_price_max": "orig_diesel_price_max",
            "diesel_price_yoy_growth": "orig_diesel_price_yoy_growth",
        }
    )
    dest_annual = annual.rename(
        columns={
            "fuel_region": "dest_padd_region",
            "diesel_price_mean": "dest_diesel_price_mean",
            "diesel_price_std": "dest_diesel_price_std",
            "diesel_price_min": "dest_diesel_price_min",
            "diesel_price_max": "dest_diesel_price_max",
            "diesel_price_yoy_growth": "dest_diesel_price_yoy_growth",
        }
    )

    merged = merged.merge(orig_annual, on=["orig_padd_region", "year"], how="left")
    merged = merged.merge(dest_annual, on=["dest_padd_region", "year"], how="left")

    merged["route_diesel_price_mean"] = merged[["orig_diesel_price_mean", "dest_diesel_price_mean"]].mean(axis=1)
    merged["route_diesel_price_gap"] = merged["dest_diesel_price_mean"] - merged["orig_diesel_price_mean"]
    merged["route_diesel_price_std_mean"] = merged[["orig_diesel_price_std", "dest_diesel_price_std"]].mean(axis=1)
    merged["route_diesel_price_yoy_growth_mean"] = merged[["orig_diesel_price_yoy_growth", "dest_diesel_price_yoy_growth"]].mean(axis=1)
    merged = merged.merge(wti, on="year", how="left")
    return merged


def main() -> None:
    ensure_dirs()
    state_lookup = load_state_lookup()
    state_lookup.to_csv(OUTPUT_DIR / "state_to_padd_lookup.csv", index=False)

    monthly = parse_diesel_monthly()
    annual = build_diesel_annual(monthly)
    wti = parse_wti_annual()

    model_df = pd.read_csv(MODEL_READY_PATH, dtype={"orig_state_fips": str, "dest_state_fips": str}, low_memory=False)
    model_df["orig_state_fips"] = model_df["orig_state_fips"].astype(str).str.zfill(2)
    model_df["dest_state_fips"] = model_df["dest_state_fips"].astype(str).str.zfill(2)
    model_with_fuel = merge_fuel(model_df, annual, wti, state_lookup)
    model_out = OUTPUT_DIR / "feature_dataset_model_ready_with_fuel.csv"
    model_with_fuel.to_csv(model_out, index=False)

    forecast_df = pd.read_csv(FORECAST_2025_PATH, dtype={"orig_state_fips": str, "dest_state_fips": str}, low_memory=False)
    forecast_df["orig_state_fips"] = forecast_df["orig_state_fips"].astype(str).str.zfill(2)
    forecast_df["dest_state_fips"] = forecast_df["dest_state_fips"].astype(str).str.zfill(2)
    forecast_with_fuel = merge_fuel(forecast_df, annual, wti, state_lookup)
    forecast_out = OUTPUT_DIR / "feature_dataset_forecast_2025_with_fuel.csv"
    forecast_with_fuel.to_csv(forecast_out, index=False)

    summary = {
        "model_ready_output": str(model_out),
        "forecast_output": str(forecast_out),
        "diesel_regions": sorted(annual["fuel_region"].unique().tolist()),
        "added_columns": [
            "orig_padd_region",
            "dest_padd_region",
            "orig_diesel_price_mean",
            "dest_diesel_price_mean",
            "route_diesel_price_mean",
            "route_diesel_price_gap",
            "route_diesel_price_std_mean",
            "route_diesel_price_yoy_growth_mean",
            "wti_price_annual",
            "wti_price_yoy_growth",
        ],
    }
    (OUTPUT_DIR / "fuel_feature_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

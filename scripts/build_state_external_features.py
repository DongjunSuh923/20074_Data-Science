from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
LOOKUP_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "lookups"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_features"

MODEL_READY_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"
FORECAST_PATH = FEATURE_DIR / "feature_dataset_forecast_2025.csv"
STATE_LOOKUP_PATH = LOOKUP_DIR / "state_lookup.csv"

SAGDP_DIR = EXTERNAL_DIR / "SAGDP(Annual GDP by State)"
CBP_DIR = EXTERNAL_DIR / "County Business Patterns 2018-2023"

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}

GDP_LINE_MAP = {
    1: "gdp_total",
    12: "gdp_manufacturing",
    34: "gdp_wholesale",
    36: "gdp_transport_warehousing",
}

CBP_CODE_MAP = {
    "------": "cbp_total",
    "31----": "cbp_mfg",
    "42----": "cbp_wholesale",
    "48----": "cbp_transport",
    "493///": "cbp_warehousing",
}


def build_state_lookup() -> pd.DataFrame:
    state_lookup = pd.read_csv(STATE_LOOKUP_PATH, dtype={"state_fips": "string"})
    state_lookup["state_fips"] = state_lookup["state_fips"].str.zfill(2)
    abbr_df = pd.DataFrame(
        {
            "state_abbr": list(STATE_ABBR_TO_FIPS.keys()),
            "state_fips": list(STATE_ABBR_TO_FIPS.values()),
        }
    )
    return abbr_df.merge(state_lookup, on="state_fips", how="left")


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"(NA)": np.nan, "NA": np.nan, "nan": np.nan, "": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def build_gdp_features() -> pd.DataFrame:
    pattern = re.compile(r"^SAGDP2_([A-Z]{2})_1997_2025\.csv$")
    frames: list[pd.DataFrame] = []
    for path in SAGDP_DIR.iterdir():
        match = pattern.match(path.name)
        if not match:
            continue
        state_abbr = match.group(1)
        if state_abbr not in STATE_ABBR_TO_FIPS:
            continue
        df = pd.read_csv(path)
        df["LineCode"] = pd.to_numeric(df["LineCode"], errors="coerce")
        df = df.loc[df["LineCode"].isin(GDP_LINE_MAP.keys())].copy()
        keep_cols = ["LineCode"] + [str(year) for year in range(2018, 2025)]
        long_df = df[keep_cols].melt(id_vars="LineCode", var_name="year", value_name="value")
        long_df["year"] = long_df["year"].astype(int)
        long_df["value"] = clean_numeric(long_df["value"])
        long_df["feature"] = long_df["LineCode"].astype(int).map(GDP_LINE_MAP)
        pivot = long_df.pivot_table(index="year", columns="feature", values="value", aggfunc="first").reset_index()
        pivot["state_abbr"] = state_abbr
        pivot["state_fips"] = STATE_ABBR_TO_FIPS[state_abbr]
        frames.append(pivot)

    gdp = pd.concat(frames, ignore_index=True).sort_values(["state_fips", "year"])
    for col in ["gdp_total", "gdp_manufacturing", "gdp_wholesale", "gdp_transport_warehousing"]:
        gdp[f"{col}_growth"] = gdp.groupby("state_fips")[col].pct_change()
    gdp["gdp_mfg_share"] = gdp["gdp_manufacturing"] / gdp["gdp_total"]
    gdp["gdp_wholesale_share"] = gdp["gdp_wholesale"] / gdp["gdp_total"]
    gdp["gdp_transport_warehousing_share"] = gdp["gdp_transport_warehousing"] / gdp["gdp_total"]
    return gdp


def build_cbp_features() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(2018, 2024):
        path = CBP_DIR / f"cbp{str(year)[2:]}st.txt"
        df = pd.read_csv(
            path,
            usecols=["fipstate", "naics", "lfo", "emp", "ap", "est"],
            dtype={"fipstate": "string", "naics": "string", "lfo": "string"},
            low_memory=False,
        )
        df["fipstate"] = df["fipstate"].str.zfill(2)
        df["naics"] = df["naics"].str.strip()
        df["lfo"] = df["lfo"].str.strip()
        df = df.loc[(df["naics"].isin(CBP_CODE_MAP.keys())) & (df["lfo"] == "-")].copy()
        df["year"] = year
        df["emp"] = clean_numeric(df["emp"])
        df["ap"] = clean_numeric(df["ap"])
        df["est"] = clean_numeric(df["est"])
        df["feature"] = df["naics"].map(CBP_CODE_MAP)

        pieces = []
        for value_col, suffix in [("emp", "emp"), ("ap", "ap"), ("est", "est")]:
            tmp = (
                df.pivot_table(index=["fipstate", "year"], columns="feature", values=value_col, aggfunc="first")
                .reset_index()
                .rename(columns=lambda c: f"{c}_{suffix}" if c in CBP_CODE_MAP.values() else c)
            )
            pieces.append(tmp)

        year_df = pieces[0]
        for piece in pieces[1:]:
            year_df = year_df.merge(piece, on=["fipstate", "year"], how="outer")
        frames.append(year_df)

    cbp = pd.concat(frames, ignore_index=True)
    cbp = cbp.rename(columns={"fipstate": "state_fips"}).sort_values(["state_fips", "year"])
    growth_cols = [col for col in cbp.columns if col not in {"state_fips", "year"}]
    for col in growth_cols:
        cbp[f"{col}_growth"] = cbp.groupby("state_fips")[col].pct_change()
    return cbp


def merge_state_features(df: pd.DataFrame, state_features: pd.DataFrame, carry_forward: bool = False) -> pd.DataFrame:
    working = df.copy()
    if carry_forward:
        last_year = int(state_features["year"].max())
        next_year = last_year + 1
        latest = state_features.loc[state_features["year"] == last_year].copy()
        latest["year"] = next_year
        state_features = pd.concat([state_features, latest], ignore_index=True)

    orig = state_features.add_prefix("orig_").rename(columns={"orig_state_fips": "orig_state_fips", "orig_year": "year"})
    dest = state_features.add_prefix("dest_").rename(columns={"dest_state_fips": "dest_state_fips", "dest_year": "year"})
    merged = working.merge(orig, on=["orig_state_fips", "year"], how="left")
    merged = merged.merge(dest, on=["dest_state_fips", "year"], how="left")

    merged["route_gdp_total_sum"] = merged["orig_gdp_total"] + merged["dest_gdp_total"]
    merged["route_gdp_total_gap"] = (merged["orig_gdp_total"] - merged["dest_gdp_total"]).abs()
    merged["route_cbp_total_emp_sum"] = merged["orig_cbp_total_emp"] + merged["dest_cbp_total_emp"]
    merged["route_cbp_warehousing_emp_sum"] = merged["orig_cbp_warehousing_emp"] + merged["dest_cbp_warehousing_emp"]
    merged["route_cbp_transport_emp_sum"] = merged["orig_cbp_transport_emp"] + merged["dest_cbp_transport_emp"]
    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    state_lookup = build_state_lookup()
    gdp = build_gdp_features()
    cbp = build_cbp_features()
    state_features = state_lookup.merge(gdp, on="state_fips", how="left").merge(cbp, on=["state_fips", "year"], how="left")
    state_features = state_features.sort_values(["state_fips", "year"]).reset_index(drop=True)
    state_features.to_csv(OUTPUT_DIR / "state_external_features.csv", index=False)

    model_ready = pd.read_csv(
        MODEL_READY_PATH,
        dtype={"orig_state_fips": "string", "dest_state_fips": "string"},
        low_memory=False,
    )
    model_ready["orig_state_fips"] = model_ready["orig_state_fips"].str.zfill(2)
    model_ready["dest_state_fips"] = model_ready["dest_state_fips"].str.zfill(2)
    model_ready_with_state = merge_state_features(model_ready, state_features, carry_forward=False)
    model_ready_path = OUTPUT_DIR / "feature_dataset_model_ready_with_state_external.csv"
    model_ready_with_state.to_csv(model_ready_path, index=False)

    forecast = pd.read_csv(
        FORECAST_PATH,
        dtype={"orig_state_fips": "string", "dest_state_fips": "string"},
        low_memory=False,
    )
    forecast["orig_state_fips"] = forecast["orig_state_fips"].str.zfill(2)
    forecast["dest_state_fips"] = forecast["dest_state_fips"].str.zfill(2)
    forecast_with_state = merge_state_features(forecast, state_features, carry_forward=True)
    forecast_path = OUTPUT_DIR / "feature_dataset_forecast_2025_with_state_external.csv"
    forecast_with_state.to_csv(forecast_path, index=False)

    summary = {
        "state_feature_file": str(OUTPUT_DIR / "state_external_features.csv"),
        "model_ready_output": str(model_ready_path),
        "forecast_output": str(forecast_path),
        "gdp_columns": [col for col in state_features.columns if col.startswith("gdp_")],
        "cbp_columns": [col for col in state_features.columns if col.startswith("cbp_")],
        "states": int(state_features["state_fips"].nunique()),
        "years": sorted(state_features["year"].dropna().unique().tolist()),
    }
    (OUTPUT_DIR / "state_external_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

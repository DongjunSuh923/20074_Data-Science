from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DIR = PROJECT_ROOT / "external_data"
FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_features"

MODEL_READY_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"
FORECAST_PATH = FEATURE_DIR / "feature_dataset_forecast_2025.csv"

SERIES_FILES = {
    "tsi_freight": ("Freight Transportation Services Index (TSIFRGHT).csv", "TSIFRGHT"),
    "trucking_price": (
        "Producer Price Index by Industry General Freight Trucking Long-Distance Truckload (PCU484121484121).csv",
        "PCU484121484121",
    ),
    "business_inventories": ("Total Business Inventories (BUSINV).csv", "BUSINV"),
    "retail_sales": ("Advance Retail Sales Retail Trade (RSXFS).csv", "RSXFS"),
}


def read_series(file_name: str, value_col: str, prefix: str) -> pd.DataFrame:
    path = EXTERNAL_DIR / file_name
    df = pd.read_csv(path)
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["year"] = df["observation_date"].dt.year
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    annual = (
        df.groupby("year")[value_col]
        .agg(mean="mean", std="std")
        .reset_index()
    )
    annual = annual.rename(
        columns={
            "mean": f"{prefix}_annual_mean",
            "std": f"{prefix}_annual_std",
        }
    )
    annual[f"{prefix}_annual_yoy_growth"] = annual[f"{prefix}_annual_mean"].pct_change()
    return annual


def build_macro_table() -> pd.DataFrame:
    annual_tables = []
    for prefix, (file_name, value_col) in SERIES_FILES.items():
        annual_tables.append(read_series(file_name, value_col, prefix))
    macro = annual_tables[0]
    for table in annual_tables[1:]:
        macro = macro.merge(table, on="year", how="outer")
    macro = macro.sort_values("year").reset_index(drop=True)
    return macro


def attach_macro(df: pd.DataFrame, macro: pd.DataFrame, carry_forward: bool = False) -> pd.DataFrame:
    merged = df.merge(macro, on="year", how="left")
    if carry_forward:
        macro_ffill = macro.sort_values("year").ffill()
        last_row = macro_ffill.iloc[-1].copy()
        next_year = int(last_row["year"]) + 1
        last_row["year"] = next_year
        macro_extended = pd.concat([macro_ffill, pd.DataFrame([last_row])], ignore_index=True)
        merged = df.merge(macro_extended, on="year", how="left")
    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    macro = build_macro_table()
    macro.to_csv(OUTPUT_DIR / "macro_annual_features.csv", index=False)

    model_ready = pd.read_csv(MODEL_READY_PATH, low_memory=False)
    model_ready_with_macro = attach_macro(model_ready, macro, carry_forward=False)
    model_ready_path = OUTPUT_DIR / "feature_dataset_model_ready_with_macro.csv"
    model_ready_with_macro.to_csv(model_ready_path, index=False)

    forecast = pd.read_csv(FORECAST_PATH, low_memory=False)
    forecast_with_macro = attach_macro(forecast, macro, carry_forward=True)
    forecast_path = OUTPUT_DIR / "feature_dataset_forecast_2025_with_macro.csv"
    forecast_with_macro.to_csv(forecast_path, index=False)

    summary = {
        "macro_feature_file": str(OUTPUT_DIR / "macro_annual_features.csv"),
        "model_ready_output": str(model_ready_path),
        "forecast_output": str(forecast_path),
        "macro_columns": [col for col in macro.columns if col != "year"],
        "years_available": macro["year"].tolist(),
    }
    (OUTPUT_DIR / "macro_feature_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

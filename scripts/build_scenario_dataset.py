from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "outputs" / "external_features" / "feature_dataset_model_ready_with_state_external.csv"
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
    "split",
]

SUM_COLS = [
    "target_tons",
]

FIRST_COLS = [
    "year_index",
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(
        INPUT_PATH,
        dtype={
            "orig_state_fips": "string",
            "dest_state_fips": "string",
            "sctg2": "string",
            "dist_band": "string",
            "trade_type": "string",
            "split": "string",
        },
        usecols=GROUP_COLS + SUM_COLS + FIRST_COLS,
        low_memory=False,
    )

    agg_map = {col: "sum" for col in SUM_COLS}
    agg_map.update({col: "first" for col in FIRST_COLS})
    scenario_df = (
        df.groupby(GROUP_COLS, dropna=False, as_index=False)
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

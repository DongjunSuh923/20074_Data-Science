from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for dep_dir in [PROJECT_ROOT / ".pydeps_fix", PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor

INPUT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "joeun_feature_eval" / "scenario_dataset_with_joeun_candidates.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "retrain_2018_train"

CATEGORICAL = ["orig_state_fips", "dest_state_fips", "sctg2", "dist_band", "trade_type"]

FEATURES = {
    "plus_accidents": [
        "route_gdp_total_gap",
        "route_gdp_per_capita_gap",
        "route_population_gap",
        "orig_gdp_per_capita",
        "dest_gdp_per_capita",
        "orig_population",
        "orig_cbp_mfg_emp",
        "dest_cbp_mfg_emp",
        "orig_gdp_wholesale_share",
        "dest_gdp_wholesale_share",
        "orig_highway_accidents",
        "dest_highway_accidents",
        "route_highway_accident_gap",
        "route_highway_accident_sum",
    ],
    "plus_both": [
        "route_gdp_total_gap",
        "route_gdp_per_capita_gap",
        "route_population_gap",
        "orig_gdp_per_capita",
        "dest_gdp_per_capita",
        "orig_population",
        "orig_cbp_mfg_emp",
        "dest_cbp_mfg_emp",
        "orig_gdp_wholesale_share",
        "dest_gdp_wholesale_share",
        "orig_real_mfg_gdp",
        "dest_real_mfg_gdp",
        "route_real_mfg_gdp_gap",
        "route_real_mfg_gdp_sum",
        "orig_highway_accidents",
        "dest_highway_accidents",
        "route_highway_accident_gap",
        "route_highway_accident_sum",
    ],
}


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, dtype={col: "string" for col in CATEGORICAL + ["split"]}, low_memory=False)


def build_rf(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ],
        sparse_threshold=0.2,
    )
    model = RandomForestRegressor(n_estimators=150, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42)
    return Pipeline([("preprocessor", pre), ("model", model)])


def build_xgb(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ]
    )
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["random_forest", "xgboost"], required=True)
    parser.add_argument("--feature-set", choices=["plus_accidents", "plus_both"], required=True)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{args.model}_{args.feature_set}.json"

    df = load_data()
    train_df = df.loc[df["split"] == "train"].copy()
    features = CATEGORICAL + FEATURES[args.feature_set]
    y_train = train_df["target_tons"].to_numpy()

    builder = build_rf if args.model == "random_forest" else build_xgb
    start = time.time()
    pipe = builder(FEATURES[args.feature_set])
    pipe.fit(train_df[features], y_train)

    rows = []
    for split_name in ["validation", "test_2023", "test_2024"]:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = np.clip(pipe.predict(split_df[features]), a_min=0, a_max=None)
        y_true = split_df["target_tons"].to_numpy()
        rows.append(
            {
                "model_name": args.model,
                "feature_set": args.feature_set,
                "split": split_name,
                "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
                "mae": float(mean_absolute_error(y_true, pred)),
                "r2": float(r2_score(y_true, pred)),
            }
        )

    payload = {
        "elapsed_seconds": time.time() - start,
        "results": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

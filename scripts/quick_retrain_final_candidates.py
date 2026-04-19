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
RESULTS_PATH = OUTPUT_DIR / "final_candidate_retrain_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "final_candidate_retrain_summary.json"

CATEGORICAL = ["orig_state_fips", "dest_state_fips", "sctg2", "dist_band", "trade_type"]

FEATURE_SETS = {
    "baseline_rf_population": [
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
    ],
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

RUNS = [
    ("random_forest", "baseline_rf_population"),
    ("random_forest", "plus_both"),
    ("xgboost", "baseline_rf_population"),
    ("xgboost", "plus_accidents"),
    ("xgboost", "plus_both"),
]


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, dtype={col: "string" for col in CATEGORICAL + ["split"]}, low_memory=False)


def build_rf(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CATEGORICAL,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ],
        sparse_threshold=0.2,
    )
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=16,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    return Pipeline([("preprocessor", pre), ("model", model)])


def build_xgb(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL,
            ),
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


def metric_row(model_name: str, feature_set: str, split_name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    df = load_data()
    train_df = df.loc[df["split"] == "train"].copy()
    y_train = train_df["target_tons"].to_numpy()

    builders = {
        "random_forest": build_rf,
        "xgboost": build_xgb,
    }

    rows: list[dict] = []
    for model_name, feature_set in RUNS:
        numeric_features = FEATURE_SETS[feature_set]
        feature_cols = CATEGORICAL + numeric_features
        pipe = builders[model_name](numeric_features)
        pipe.fit(train_df[feature_cols], y_train)

        for split_name in ["validation", "test_2023", "test_2024"]:
            split_df = df.loc[df["split"] == split_name].copy()
            pred = np.clip(pipe.predict(split_df[feature_cols]), a_min=0, a_max=None)
            rows.append(metric_row(model_name, feature_set, split_name, split_df["target_tons"].to_numpy(), pred))

    results = pd.DataFrame(rows).sort_values(["split", "rmse"])
    results.to_csv(RESULTS_PATH, index=False)

    summary = {
        "results_path": str(RESULTS_PATH),
        "elapsed_seconds": time.time() - start,
        "best_by_split": results.groupby("split", as_index=False).first().to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

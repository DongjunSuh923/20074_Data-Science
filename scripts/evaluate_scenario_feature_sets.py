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


INPUT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features" / "scenario_dataset_with_population.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "feature_set_selection"
RESULTS_PATH = OUTPUT_DIR / "feature_set_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "feature_set_summary.md"
SUMMARY_JSON = OUTPUT_DIR / "feature_set_summary.json"

CATEGORICAL = [
    "orig_state_fips",
    "dest_state_fips",
    "sctg2",
    "dist_band",
    "trade_type",
]

TARGET = "target_tons"
SPLITS = ["validation", "test_2023", "test_2024"]

FEATURE_SETS = {
    "common_compact": [
        "route_gdp_total_gap",
        "orig_gdp_wholesale",
        "orig_cbp_transport_emp",
        "orig_cbp_warehousing_emp",
        "orig_cbp_mfg_emp",
        "route_cbp_total_emp_sum",
    ],
    "xgb_compact": [
        "route_gdp_total_gap",
        "orig_gdp_wholesale",
        "orig_gdp_wholesale_share",
        "orig_cbp_transport_emp",
        "orig_cbp_warehousing_emp",
        "orig_cbp_mfg_emp",
        "route_cbp_total_emp_sum",
        "route_cbp_transport_emp_sum",
        "route_cbp_warehousing_emp_sum",
    ],
    "rf_compact": [
        "route_gdp_total_gap",
        "orig_cbp_mfg_emp",
        "dest_cbp_mfg_emp",
        "orig_gdp_wholesale_share",
        "dest_gdp_wholesale_share",
        "orig_cbp_warehousing_est",
        "route_cbp_warehousing_emp_sum",
        "orig_cbp_transport_emp",
        "dest_cbp_transport_emp",
    ],
    "rf_population": [
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
    "full_with_population": [
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
        "orig_population",
        "dest_population",
        "orig_population_growth",
        "dest_population_growth",
        "orig_gdp_per_capita",
        "dest_gdp_per_capita",
        "route_population_sum",
        "route_population_gap",
        "route_gdp_per_capita_gap",
    ],
}


def load_data() -> pd.DataFrame:
    dtype_map = {col: "string" for col in CATEGORICAL + ["split"]}
    return pd.read_csv(INPUT_PATH, dtype=dtype_map, low_memory=False)


def build_rf(numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
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
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_xgb(numeric_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
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
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def metric_row(model_name: str, feature_set: str, split_name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
        "feature_count": len(CATEGORICAL) + len(FEATURE_SETS[feature_set]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y_true)) ** 2))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    df = load_data()
    train_df = df.loc[df["split"] == "train"].copy()

    rows: list[dict] = []
    model_builders = {
        "xgboost": build_xgb,
        "random_forest": build_rf,
    }

    for feature_set_name, numeric_features in FEATURE_SETS.items():
        feature_cols = CATEGORICAL + numeric_features
        X_train = train_df[feature_cols]
        y_train = train_df[TARGET].to_numpy()

        for model_name, builder in model_builders.items():
            pipeline = builder(numeric_features)
            pipeline.fit(X_train, y_train)

            for split_name in SPLITS:
                split_df = df.loc[df["split"] == split_name].copy()
                pred = pipeline.predict(split_df[feature_cols])
                pred = np.clip(pred, a_min=0, a_max=None)
                rows.append(metric_row(model_name, feature_set_name, split_name, split_df[TARGET].to_numpy(), pred))

    results_df = pd.DataFrame(rows).sort_values(["split", "rmse", "model_name", "feature_set"])
    results_df.to_csv(RESULTS_PATH, index=False)

    best_by_split = results_df.groupby("split", as_index=False).first()
    best_by_model = (
        results_df.sort_values(["model_name", "rmse"])
        .groupby(["model_name", "split"], as_index=False)
        .first()
    )

    lines = [
        "# Scenario Feature Set Selection",
        "",
        "- Models compared: XGBoost, Random Forest",
        "- Target: raw tons",
        "- Unit: lag-free state-state x commodity x dist_band x trade_type x year",
        "",
        "## Best By Split",
    ]
    for _, row in best_by_split.iterrows():
        lines.append(
            f"- {row['split']}: {row['model_name']} / {row['feature_set']} "
            f"(RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}, features={int(row['feature_count'])})"
        )
    lines += ["", "## Best By Model And Split"]
    for _, row in best_by_model.iterrows():
        lines.append(
            f"- {row['model_name']} / {row['split']}: {row['feature_set']} "
            f"(RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f})"
        )
    lines += ["", f"Elapsed seconds: {time.time() - start:.2f}"]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "results_path": str(RESULTS_PATH),
        "best_by_split": best_by_split.to_dict(orient="records"),
        "best_by_model_and_split": best_by_model.to_dict(orient="records"),
        "elapsed_seconds": time.time() - start,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from xgboost import XGBRegressor


INPUT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features" / "scenario_dataset_with_population.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features"
RESULTS_PATH = OUTPUT_DIR / "scenario_model_results_with_population.csv"
SUMMARY_PATH = OUTPUT_DIR / "scenario_model_summary_with_population.md"
SUMMARY_JSON = OUTPUT_DIR / "scenario_model_summary_with_population.json"

CATEGORICAL = [
    "orig_state_fips",
    "dest_state_fips",
    "sctg2",
    "dist_band",
    "trade_type",
]

NUMERIC = [
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
]

TARGETS = {"raw": "target_tons", "log": "target_log_tons"}
SPLITS = ["validation", "test_2023", "test_2024"]


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, dtype={col: "string" for col in CATEGORICAL + ["split"]}, low_memory=False)


def build_linear() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def build_rf() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ],
        sparse_threshold=0.2,
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_xgb() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ]
    )
    model = XGBRegressor(
        n_estimators=500,
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


def build_lgbm() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ]
    )
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def metric_row(model_name: str, target_name: str, split_name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model_name": model_name,
        "target_name": target_name,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
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
    X_train = train_df[CATEGORICAL + NUMERIC]

    rows = []
    for target_name, target_col in TARGETS.items():
        y_train = train_df[target_col].to_numpy()
        for model_name, builder in {"linear": build_linear, "random_forest": build_rf, "xgboost": build_xgb, "lightgbm": build_lgbm}.items():
            pipeline = builder()
            pipeline.fit(X_train, y_train)
            for split_name in SPLITS:
                split_df = df.loc[df["split"] == split_name].copy()
                pred = pipeline.predict(split_df[CATEGORICAL + NUMERIC])
                if target_name == "log":
                    pred = np.expm1(pred)
                pred = np.clip(pred, a_min=0, a_max=None)
                y_true = split_df["target_tons"].to_numpy()
                rows.append(metric_row(model_name, target_name, split_name, y_true, pred))

    results_df = pd.DataFrame(rows).sort_values(["split", "rmse"])
    results_df.to_csv(RESULTS_PATH, index=False)

    best = results_df.groupby("split", as_index=False).first()
    lines = [
        "# Scenario Model Results With Population",
        "",
        "- Added state population and GDP-per-capita features.",
        "- Same lag-free scenario unit retained.",
        "",
        "## Best by split",
    ]
    for _, row in best.iterrows():
        lines.append(f"- {row['split']}: {row['model_name']} ({row['target_name']}) RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}")
    lines += ["", f"Elapsed seconds: {time.time() - start:.2f}"]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "results_path": str(RESULTS_PATH),
        "best_by_split": best.to_dict(orient="records"),
        "elapsed_seconds": time.time() - start,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

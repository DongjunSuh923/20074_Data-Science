from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


INPUT_PATH = PROJECT_ROOT / "outputs" / "external_features" / "feature_dataset_model_ready_with_macro.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "redesign_experiments"
RESULTS_CSV = OUTPUT_DIR / "redesign_experiment_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "redesign_experiment_summary.json"
SUMMARY_MD = OUTPUT_DIR / "redesign_experiment_summary.md"
PLOT_PATH = OUTPUT_DIR / "validation_rmse_comparison.png"

BASE_CATEGORICAL = ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"]
FEATURE_SETS = {
    "core": {
        "categorical": BASE_CATEGORICAL,
        "numeric": [
            "tons_lag1",
            "tons_lag2",
            "tons_growth_rate",
            "year_index",
        ],
    },
    "core_route": {
        "categorical": BASE_CATEGORICAL,
        "numeric": [
            "tons_lag1",
            "tons_lag2",
            "tons_growth_rate",
            "year_index",
            "orig_total_outbound_tons_lag1",
            "dest_total_inbound_tons_lag1",
            "route_share_of_origin_lag1",
            "route_share_of_dest_lag1",
            "corridor_sctg2_count_lag1",
            "tons_zero_flag_lag1",
        ],
    },
    "core_route_macro": {
        "categorical": BASE_CATEGORICAL,
        "numeric": [
            "tons_lag1",
            "tons_lag2",
            "tons_growth_rate",
            "year_index",
            "orig_total_outbound_tons_lag1",
            "dest_total_inbound_tons_lag1",
            "route_share_of_origin_lag1",
            "route_share_of_dest_lag1",
            "corridor_sctg2_count_lag1",
            "tons_zero_flag_lag1",
            "tsi_freight_annual_mean",
            "tsi_freight_annual_yoy_growth",
            "trucking_price_annual_mean",
            "trucking_price_annual_yoy_growth",
            "business_inventories_annual_mean",
            "business_inventories_annual_yoy_growth",
            "retail_sales_annual_mean",
            "retail_sales_annual_yoy_growth",
        ],
    },
}

TARGET_SPECS = {
    "raw": {"column": "target_tons", "is_log": False},
    "log": {"column": "target_log_tons", "is_log": True},
}

SPLITS = ["validation", "test_2023", "test_2024"]
TRAIN_SAMPLE_CAP = 300_000
RANDOM_STATE = 42


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    usecols = {
        "target_tons",
        "target_log_tons",
        "split",
        "year",
    }
    for config in FEATURE_SETS.values():
        usecols.update(config["categorical"])
        usecols.update(config["numeric"])
    dtype_map = {
        "sctg2": "string",
        "dist_band": "string",
        "trade_type": "string",
        "orig_state_fips": "string",
        "dest_state_fips": "string",
        "split": "string",
    }
    return pd.read_csv(INPUT_PATH, usecols=sorted(usecols), dtype=dtype_map, low_memory=False)


def metric_row(model_name: str, feature_set: str, target_spec: str, split_name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "model_name": model_name,
        "feature_set": feature_set,
        "target_spec": target_spec,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def build_linear_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)),
        ]
    )
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, cat_features),
            ("num", numeric_pipe, num_features),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def build_tree_preprocessor(cat_features: list[str], num_features: list[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, cat_features),
            ("num", numeric_pipe, num_features),
        ]
    )


def make_models() -> dict[str, object]:
    return {
        "linear": "linear",
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="regression",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        ),
    }


def inverse_target(pred: np.ndarray, is_log: bool) -> np.ndarray:
    if is_log:
        pred = np.expm1(pred)
    return np.clip(pred, a_min=0, a_max=None)


def evaluate_combo(
    df: pd.DataFrame,
    model_name: str,
    model_obj: object,
    feature_set_name: str,
    target_name: str,
) -> list[dict]:
    config = FEATURE_SETS[feature_set_name]
    cat_features = config["categorical"]
    num_features = config["numeric"]
    target_col = TARGET_SPECS[target_name]["column"]
    is_log = TARGET_SPECS[target_name]["is_log"]

    train_df = df.loc[df["split"] == "train"].copy()
    X_train = train_df[cat_features + num_features]
    y_train = train_df[target_col].to_numpy()

    rows = []

    if model_name == "linear":
        pipeline = build_linear_pipeline(cat_features, num_features)
        pipeline.fit(X_train, y_train)
        for split_name in SPLITS:
            split_df = df.loc[df["split"] == split_name].copy()
            pred = pipeline.predict(split_df[cat_features + num_features])
            pred_tons = inverse_target(pred, is_log=is_log)
            y_true = split_df["target_tons"].to_numpy()
            rows.append(metric_row(model_name, feature_set_name, target_name, split_name, y_true, pred_tons))
        return rows

    train_fit = train_df.copy()
    if train_fit.shape[0] > TRAIN_SAMPLE_CAP:
        train_fit = train_fit.sample(TRAIN_SAMPLE_CAP, random_state=RANDOM_STATE)

    preprocessor = build_tree_preprocessor(cat_features, num_features)
    X_train_tree = preprocessor.fit_transform(train_fit[cat_features + num_features])
    y_train_tree = train_fit[target_col].to_numpy()
    model_obj.fit(X_train_tree, y_train_tree)

    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        X_split = preprocessor.transform(split_df[cat_features + num_features])
        pred = np.asarray(model_obj.predict(X_split))
        pred_tons = inverse_target(pred, is_log=is_log)
        y_true = split_df["target_tons"].to_numpy()
        rows.append(metric_row(model_name, feature_set_name, target_name, split_name, y_true, pred_tons))

    return rows


def save_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.loc[results_df["split"] == "validation"].copy()
    plot_df["label"] = plot_df["model_name"] + "\n" + plot_df["feature_set"] + "\n" + plot_df["target_spec"]
    plot_df = plot_df.sort_values("rmse")
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 7))
    plt.bar(plot_df["label"], plot_df["rmse"], color="#3a86ff")
    plt.ylabel("Validation RMSE")
    plt.title("Model Redesign Experiments")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)
    plt.close()


def write_summary(results_df: pd.DataFrame, elapsed_sec: float) -> None:
    lines = [
        "# Redesign Experiments",
        "",
        "## Setup",
        "- Models kept: Linear / RF / XGB / LGBM",
        "- Target specs compared: raw tons vs log1p(tons)",
        "- Feature sets compared: core / core_route / core_route_macro",
        f"- Train sample cap for tree models: {TRAIN_SAMPLE_CAP:,}",
        "",
        "## Best Validation Results",
    ]
    best_val = results_df.loc[results_df["split"] == "validation"].sort_values("rmse").head(10)
    for _, row in best_val.iterrows():
        lines.append(
            f"- {row['model_name']} | {row['feature_set']} | {row['target_spec']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}"
        )
    lines += ["", f"Elapsed seconds: {elapsed_sec:.2f}"]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()
    models = make_models()

    rows: list[dict] = []
    for feature_set_name in FEATURE_SETS:
        for target_name in TARGET_SPECS:
            for model_name, model_obj in models.items():
                rows.extend(evaluate_combo(df, model_name, model_obj, feature_set_name, target_name))

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    save_plot(results_df)

    elapsed = time.time() - start
    best_validation = results_df.loc[results_df["split"] == "validation"].sort_values("rmse").head(10)
    summary = {
        "results_path": str(RESULTS_CSV),
        "plot_path": str(PLOT_PATH),
        "elapsed_seconds": elapsed,
        "best_validation_rows": best_validation.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(results_df, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

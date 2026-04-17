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
import joblib
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


INPUT_PATH = PROJECT_ROOT / "outputs" / "external_features" / "feature_dataset_model_ready_with_fuel.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "outputs" / "models" / "baseline_suite" / "baseline_metrics.csv"
TREE_METRICS_PATH = PROJECT_ROOT / "outputs" / "models" / "tree_models" / "tree_model_metrics.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "fuel_feature_impact"
METRICS_CSV = OUTPUT_DIR / "fuel_model_metrics.csv"
COMPARISON_CSV = OUTPUT_DIR / "fuel_vs_original_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "fuel_feature_impact_summary.json"
SUMMARY_MD = OUTPUT_DIR / "fuel_feature_impact_summary.md"
DELTA_PNG = OUTPUT_DIR / "rmse_delta_vs_original.png"

CATEGORICAL_FEATURES = [
    "sctg2",
    "dist_band",
    "trade_type",
    "orig_state_fips",
    "dest_state_fips",
    "orig_padd_region",
    "dest_padd_region",
]

NUMERIC_FEATURES = [
    "tons_lag1",
    "tons_lag2",
    "tons_growth_rate",
    "value_per_ton_lag1",
    "year_index",
    "orig_total_outbound_tons_lag1",
    "dest_total_inbound_tons_lag1",
    "orig_unique_dest_count_lag1",
    "dest_unique_orig_count_lag1",
    "route_share_of_origin_lag1",
    "route_share_of_dest_lag1",
    "corridor_sctg2_count_lag1",
    "tons_zero_flag_lag1",
    "orig_diesel_price_mean",
    "dest_diesel_price_mean",
    "route_diesel_price_mean",
    "route_diesel_price_gap",
    "route_diesel_price_std_mean",
    "route_diesel_price_yoy_growth_mean",
    "wti_price_annual",
    "wti_price_yoy_growth",
]

USECOLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [
    "target_tons",
    "split",
    "year",
]

TRAIN_SAMPLE_CAP = 400_000
RANDOM_STATE = 42
SPLITS = ["validation", "test_2023", "test_2024"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    dtype_map = {
        "sctg2": "string",
        "dist_band": "string",
        "trade_type": "string",
        "orig_state_fips": "string",
        "dest_state_fips": "string",
        "orig_padd_region": "string",
        "dest_padd_region": "string",
        "split": "string",
    }
    return pd.read_csv(INPUT_PATH, usecols=USECOLS, dtype=dtype_map, low_memory=False)


def metric_dict(model_name: str, split_name: str, split_df: pd.DataFrame, pred: np.ndarray) -> dict:
    y_true = split_df["target_tons"].to_numpy()
    return {
        "model_name": model_name,
        "split": split_name,
        "year_min": int(split_df["year"].min()),
        "year_max": int(split_df["year"].max()),
        "row_count": int(split_df.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }


def evaluate_persistence(df: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = np.clip(split_df["tons_lag1"].to_numpy(), a_min=0, a_max=None)
        rows.append(metric_dict("persistence_with_fuel_dataset", split_name, split_df, pred))
    return rows


def build_linear_pipeline() -> Pipeline:
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
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERIC_FEATURES),
        ],
        sparse_threshold=0.3,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def build_tree_preprocessor() -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("num", numeric_pipe, NUMERIC_FEATURES),
        ]
    )


def evaluate_linear(df: pd.DataFrame) -> list[dict]:
    train_df = df.loc[df["split"] == "train"].copy()
    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df["target_tons"].to_numpy()

    pipeline = build_linear_pipeline()
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, OUTPUT_DIR / "linear_with_fuel.joblib")

    rows = []
    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = np.clip(
            pipeline.predict(split_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]),
            a_min=0,
            a_max=None,
        )
        rows.append(metric_dict("linear_with_fuel", split_name, split_df, pred))
    return rows


def tree_models() -> dict[str, object]:
    return {
        "random_forest_with_fuel": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost_with_fuel": XGBRegressor(
            n_estimators=400,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "lightgbm_with_fuel": LGBMRegressor(
            n_estimators=400,
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


def evaluate_tree_model(name: str, model: object, df: pd.DataFrame) -> list[dict]:
    train_df = df.loc[df["split"] == "train"].copy()
    if train_df.shape[0] > TRAIN_SAMPLE_CAP:
        train_df = train_df.sample(TRAIN_SAMPLE_CAP, random_state=RANDOM_STATE)

    preprocessor = build_tree_preprocessor()
    X_train = preprocessor.fit_transform(train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    y_train = train_df["target_tons"].to_numpy()
    model.fit(X_train, y_train)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "categorical_features": CATEGORICAL_FEATURES,
            "numeric_features": NUMERIC_FEATURES,
            "train_sample_rows": int(train_df.shape[0]),
        },
        OUTPUT_DIR / f"{name}.joblib",
    )

    rows = []
    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = np.asarray(model.predict(preprocessor.transform(split_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])))
        pred = np.clip(pred, a_min=0, a_max=None)
        rows.append(metric_dict(name, split_name, split_df, pred))
    return rows


def load_original_metrics() -> pd.DataFrame:
    baseline = pd.read_csv(BASELINE_METRICS_PATH)
    tree = pd.read_csv(TREE_METRICS_PATH)
    keep_baseline = baseline.loc[
        baseline["model_name"].isin(["persistence_tons_lag1", "linear_full"]),
        ["model_name", "split", "rmse", "mae", "r2"],
    ].copy()
    keep_tree = tree[["model_name", "split", "rmse", "mae", "r2"]].copy()
    original = pd.concat([keep_baseline, keep_tree], ignore_index=True)
    mapping = {
        "persistence_tons_lag1": "persistence_with_fuel_dataset",
        "linear_full": "linear_with_fuel",
        "random_forest": "random_forest_with_fuel",
        "xgboost": "xgboost_with_fuel",
        "lightgbm": "lightgbm_with_fuel",
    }
    original["fuel_model_name"] = original["model_name"].map(mapping)
    return original


def save_delta_plot(comparison_df: pd.DataFrame) -> None:
    plot_df = comparison_df.copy()
    plot_df["split_model"] = plot_df["fuel_model_name"] + "\n" + plot_df["split"]
    plot_df = plot_df.sort_values(["fuel_model_name", "split"])
    plt.style.use("ggplot")
    plt.figure(figsize=(11, 6))
    colors = ["#2a9d8f" if x < 0 else "#e76f51" for x in plot_df["rmse_delta"]]
    plt.bar(plot_df["split_model"], plot_df["rmse_delta"], color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("RMSE Delta (Fuel - Original)")
    plt.title("Fuel Feature Impact on RMSE")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(DELTA_PNG, dpi=180)
    plt.close()


def write_summary(metrics_df: pd.DataFrame, comparison_df: pd.DataFrame, elapsed_sec: float) -> None:
    lines = [
        "# Fuel Feature Impact",
        "",
        "## Setup",
        "- Input dataset: `feature_dataset_model_ready_with_fuel.csv`",
        "- Added fuel features: origin/destination/route diesel statistics + annual WTI",
        "- Split policy: train=2020-2021, validation=2022, test_2023=2023, test_2024=2024",
        f"- Tree-model train sample cap: {TRAIN_SAMPLE_CAP:,}",
        "",
        "## Fuel-Enhanced Metrics",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['model_name']} / {row['split']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}"
        )
    lines += [
        "",
        "## Comparison With Original Models",
    ]
    for _, row in comparison_df.iterrows():
        direction = "improved" if row["rmse_delta"] < 0 else "worsened"
        lines.append(
            f"- {row['fuel_model_name']} / {row['split']}: RMSE delta={row['rmse_delta']:.4f} ({direction}), MAE delta={row['mae_delta']:.4f}, R2 delta={row['r2_delta']:.4f}"
        )
    lines += [
        "",
        f"Elapsed seconds: {elapsed_sec:.2f}",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()

    metric_rows = []
    metric_rows.extend(evaluate_persistence(df))
    metric_rows.extend(evaluate_linear(df))
    for name, model in tree_models().items():
        metric_rows.extend(evaluate_tree_model(name, model, df))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    original_df = load_original_metrics()
    comparison_df = metrics_df.merge(
        original_df,
        left_on=["model_name", "split"],
        right_on=["fuel_model_name", "split"],
        how="left",
        suffixes=("_fuel", "_original"),
    )
    comparison_df["rmse_delta"] = comparison_df["rmse_fuel"] - comparison_df["rmse_original"]
    comparison_df["mae_delta"] = comparison_df["mae_fuel"] - comparison_df["mae_original"]
    comparison_df["r2_delta"] = comparison_df["r2_fuel"] - comparison_df["r2_original"]
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    save_delta_plot(comparison_df)

    elapsed = time.time() - start
    summary = {
        "metrics_path": str(METRICS_CSV),
        "comparison_path": str(COMPARISON_CSV),
        "delta_plot_path": str(DELTA_PNG),
        "elapsed_seconds": elapsed,
        "best_validation_rmse_with_fuel": metrics_df.loc[metrics_df["split"] == "validation"].sort_values("rmse").iloc[0].to_dict(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(metrics_df, comparison_df, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

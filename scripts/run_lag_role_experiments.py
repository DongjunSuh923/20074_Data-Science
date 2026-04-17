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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = PROJECT_ROOT / "outputs" / "external_features" / "feature_dataset_model_ready_with_macro.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "lag_role_experiments"
RESULTS_CSV = OUTPUT_DIR / "lag_role_results.csv"
SUMMARY_JSON = OUTPUT_DIR / "lag_role_summary.json"
SUMMARY_MD = OUTPUT_DIR / "lag_role_summary.md"
PLOT_PATH = OUTPUT_DIR / "lag_role_validation_rmse.png"

SPLITS = ["validation", "test_2023", "test_2024"]

EXPERIMENTS = {
    "persistence": {
        "type": "baseline",
    },
    "lag_only_linear": {
        "type": "linear",
        "categorical": [],
        "numeric": ["tons_lag1", "tons_lag2", "tons_growth_rate"],
        "target": "target_tons",
        "reconstruct_from_delta": False,
    },
    "no_lag_linear": {
        "type": "linear",
        "categorical": ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"],
        "numeric": [
            "year_index",
            "tsi_freight_annual_mean",
            "tsi_freight_annual_yoy_growth",
            "trucking_price_annual_mean",
            "trucking_price_annual_yoy_growth",
            "business_inventories_annual_mean",
            "business_inventories_annual_yoy_growth",
            "retail_sales_annual_mean",
            "retail_sales_annual_yoy_growth",
        ],
        "target": "target_tons",
        "reconstruct_from_delta": False,
    },
    "full_linear": {
        "type": "linear",
        "categorical": ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"],
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
        "target": "target_tons",
        "reconstruct_from_delta": False,
    },
    "delta_nonlag_linear": {
        "type": "linear",
        "categorical": ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"],
        "numeric": [
            "year_index",
            "tsi_freight_annual_mean",
            "tsi_freight_annual_yoy_growth",
            "trucking_price_annual_mean",
            "trucking_price_annual_yoy_growth",
            "business_inventories_annual_mean",
            "business_inventories_annual_yoy_growth",
            "retail_sales_annual_mean",
            "retail_sales_annual_yoy_growth",
        ],
        "target": "target_delta",
        "reconstruct_from_delta": True,
    },
    "delta_with_lag_linear": {
        "type": "linear",
        "categorical": ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"],
        "numeric": [
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
        "target": "target_delta",
        "reconstruct_from_delta": True,
    },
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def required_columns() -> list[str]:
    cols = {"target_tons", "split", "year", "tons_lag1"}
    for config in EXPERIMENTS.values():
        if config["type"] != "linear":
            continue
        cols.update(config["categorical"])
        cols.update(config["numeric"])
    return sorted(cols)


def load_data() -> pd.DataFrame:
    dtype_map = {
        "sctg2": "string",
        "dist_band": "string",
        "trade_type": "string",
        "orig_state_fips": "string",
        "dest_state_fips": "string",
        "split": "string",
    }
    df = pd.read_csv(INPUT_PATH, usecols=required_columns(), dtype=dtype_map, low_memory=False)
    df["target_delta"] = df["target_tons"] - df["tons_lag1"]
    return df


def build_pipeline(cat_features: list[str], num_features: list[str]) -> Pipeline:
    transformers = []
    if cat_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)),
                    ]
                ),
                cat_features,
            )
        )
    if num_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def metric_row(experiment_name: str, split_name: str, y_true: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "experiment_name": experiment_name,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(y_true)) ** 2))),
    }


def run_persistence(df: pd.DataFrame) -> list[dict]:
    rows = []
    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = np.clip(split_df["tons_lag1"].to_numpy(), a_min=0, a_max=None)
        y_true = split_df["target_tons"].to_numpy()
        rows.append(metric_row("persistence", split_name, y_true, pred))
    return rows


def run_linear(df: pd.DataFrame, experiment_name: str, config: dict) -> list[dict]:
    cat_features = config["categorical"]
    num_features = config["numeric"]
    target_col = config["target"]
    reconstruct_from_delta = config["reconstruct_from_delta"]

    train_df = df.loc[df["split"] == "train"].copy()
    X_train = train_df[cat_features + num_features]
    y_train = train_df[target_col].to_numpy()

    pipeline = build_pipeline(cat_features, num_features)
    pipeline.fit(X_train, y_train)

    rows = []
    for split_name in SPLITS:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = pipeline.predict(split_df[cat_features + num_features])
        if reconstruct_from_delta:
            pred = split_df["tons_lag1"].to_numpy() + pred
        pred = np.clip(pred, a_min=0, a_max=None)
        y_true = split_df["target_tons"].to_numpy()
        rows.append(metric_row(experiment_name, split_name, y_true, pred))
    return rows


def save_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.loc[results_df["split"] == "validation"].sort_values("rmse").copy()
    plt.style.use("ggplot")
    plt.figure(figsize=(11, 6))
    plt.bar(plot_df["experiment_name"], plot_df["rmse"], color="#3a86ff")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Validation RMSE")
    plt.title("Lag Role Experiments")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)
    plt.close()


def write_summary(results_df: pd.DataFrame, elapsed_sec: float) -> None:
    lines = [
        "# Lag Role Experiments",
        "",
        "## Purpose",
        "- Check whether the current model behaves like a restoration model dominated by lag features.",
        "- Compare lag-only, no-lag, and delta-target formulations.",
        "",
        "## Results",
    ]
    for split_name in ["validation", "test_2023", "test_2024"]:
        subset = results_df.loc[results_df["split"] == split_name].sort_values("rmse")
        lines.append(f"- {split_name}:")
        for _, row in subset.iterrows():
            lines.append(
                f"  - {row['experiment_name']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, RMSLE={row['rmsle']:.4f}, R2={row['r2']:.4f}"
            )
    lines += ["", f"Elapsed seconds: {elapsed_sec:.2f}"]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()

    rows: list[dict] = []
    rows.extend(run_persistence(df))
    for experiment_name, config in EXPERIMENTS.items():
        if config["type"] != "linear":
            continue
        rows.extend(run_linear(df, experiment_name, config))

    results_df = pd.DataFrame(rows).sort_values(["split", "rmse", "experiment_name"]).reset_index(drop=True)
    results_df.to_csv(RESULTS_CSV, index=False)
    save_plot(results_df)

    elapsed = time.time() - start
    summary = {
        "results_path": str(RESULTS_CSV),
        "plot_path": str(PLOT_PATH),
        "elapsed_seconds": elapsed,
        "best_by_split": {
            split_name: results_df.loc[results_df["split"] == split_name].sort_values("rmse").iloc[0].to_dict()
            for split_name in SPLITS
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(results_df, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

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
from lightgbm import LGBMRegressor, early_stopping as lgb_early_stopping
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor


INPUT_PATH = PROJECT_ROOT / "outputs" / "external_features" / "feature_dataset_model_ready_with_macro.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "tree_revalidation"
RESULTS_CSV = OUTPUT_DIR / "tree_revalidation_results.csv"
SEGMENT_CSV = OUTPUT_DIR / "tree_revalidation_segment_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "tree_revalidation_summary.json"
SUMMARY_MD = OUTPUT_DIR / "tree_revalidation_summary.md"
PLOT_PATH = OUTPUT_DIR / "tree_revalidation_validation_rmse.png"

CATEGORICAL_FEATURES = ["sctg2", "dist_band", "trade_type", "orig_state_fips", "dest_state_fips"]
NUMERIC_FEATURES = [
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
]
USECOLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["target_tons", "target_log_tons", "split", "year"]
TARGETS = {"raw": "target_tons", "log": "target_log_tons"}
SPLITS = ["validation", "test_2023", "test_2024"]
RANDOM_STATE = 42


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    dtype_map = {
        "sctg2": "string",
        "dist_band": "string",
        "trade_type": "string",
        "orig_state_fips": "string",
        "dest_state_fips": "string",
        "split": "string",
    }
    return pd.read_csv(INPUT_PATH, usecols=USECOLS, dtype=dtype_map, low_memory=False)


def inverse_target(pred: np.ndarray, target_name: str) -> np.ndarray:
    if target_name == "log":
        pred = np.expm1(pred)
    return np.clip(pred, a_min=0, a_max=None)


def base_metrics(y_true: np.ndarray, pred_tons: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred_tons))),
        "mae": float(mean_absolute_error(y_true, pred_tons)),
        "r2": float(r2_score(y_true, pred_tons)),
        "rmsle": float(np.sqrt(np.mean((np.log1p(pred_tons) - np.log1p(y_true)) ** 2))),
    }


def append_segment_metrics(rows: list[dict], experiment_name: str, split_name: str, y_true: np.ndarray, pred_tons: np.ndarray) -> None:
    thresholds = {
        "all": np.ones_like(y_true, dtype=bool),
        "top10pct_true": y_true >= np.quantile(y_true, 0.9),
        "bottom90pct_true": y_true < np.quantile(y_true, 0.9),
        "positive_true": y_true > 0,
    }
    for segment_name, mask in thresholds.items():
        if mask.sum() == 0:
            continue
        metrics = base_metrics(y_true[mask], pred_tons[mask])
        rows.append(
            {
                "experiment_name": experiment_name,
                "split": split_name,
                "segment": segment_name,
                "row_count": int(mask.sum()),
                **metrics,
            }
        )


def metric_row(experiment_name: str, split_name: str, y_true: np.ndarray, pred_tons: np.ndarray) -> dict:
    return {
        "experiment_name": experiment_name,
        "split": split_name,
        "row_count": int(y_true.shape[0]),
        **base_metrics(y_true, pred_tons),
    }


def ordinal_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ]
    )


def onehot_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ],
        sparse_threshold=0.2,
    )


def fit_predict_tree(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_dfs: dict[str, pd.DataFrame],
    target_name: str,
    experiment_name: str,
    model: object,
    preprocessor: ColumnTransformer,
    use_early_stopping: bool = False,
) -> tuple[list[dict], list[dict]]:
    target_col = TARGETS[target_name]
    X_train = preprocessor.fit_transform(train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    y_train = train_df[target_col].to_numpy()

    fit_kwargs = {}
    if use_early_stopping:
        X_val = preprocessor.transform(validation_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
        y_val = validation_df[target_col].to_numpy()
        if isinstance(model, XGBRegressor):
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "verbose": False,
            }
            model.set_params(early_stopping_rounds=50)
        elif isinstance(model, LGBMRegressor):
            fit_kwargs = {
                "eval_set": [(X_val, y_val)],
                "eval_metric": "l2",
                "callbacks": [lgb_early_stopping(50, verbose=False)],
            }

    model.fit(X_train, y_train, **fit_kwargs)

    metric_rows: list[dict] = []
    segment_rows: list[dict] = []
    for split_name, split_df in test_dfs.items():
        X_split = preprocessor.transform(split_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
        pred = np.asarray(model.predict(X_split))
        pred_tons = inverse_target(pred, target_name)
        y_true = split_df["target_tons"].to_numpy()
        metric_rows.append(metric_row(experiment_name, split_name, y_true, pred_tons))
        append_segment_metrics(segment_rows, experiment_name, split_name, y_true, pred_tons)
    return metric_rows, segment_rows


def make_fulltrain_models(target_name: str) -> dict[str, tuple[object, ColumnTransformer, bool]]:
    return {
        f"rf_fulltrain_{target_name}_ordinal": (
            RandomForestRegressor(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            ordinal_preprocessor(),
            False,
        ),
        f"xgb_fulltrain_{target_name}_ordinal": (
            XGBRegressor(
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
            ordinal_preprocessor(),
            False,
        ),
        f"lgbm_fulltrain_{target_name}_ordinal": (
            LGBMRegressor(
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
            ordinal_preprocessor(),
            False,
        ),
    }


def make_encoding_models(target_name: str) -> dict[str, tuple[object, ColumnTransformer, bool]]:
    return {
        f"xgb_fulltrain_{target_name}_onehot": (
            XGBRegressor(
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
            onehot_preprocessor(),
            False,
        ),
        f"lgbm_fulltrain_{target_name}_onehot": (
            LGBMRegressor(
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
            onehot_preprocessor(),
            False,
        ),
    }


def make_tuned_models(target_name: str) -> dict[str, tuple[object, ColumnTransformer, bool]]:
    return {
        f"xgb_tuned_{target_name}_onehot_es": (
            XGBRegressor(
                n_estimators=1200,
                max_depth=6,
                learning_rate=0.03,
                min_child_weight=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                reg_alpha=0.2,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            onehot_preprocessor(),
            True,
        ),
        f"lgbm_tuned_{target_name}_onehot_es": (
            LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=95,
                min_child_samples=60,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=2.0,
                objective="regression",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=-1,
            ),
            onehot_preprocessor(),
            True,
        ),
    }


def save_plot(results_df: pd.DataFrame) -> None:
    plot_df = results_df.loc[results_df["split"] == "validation"].sort_values("rmse").copy()
    plt.style.use("ggplot")
    plt.figure(figsize=(13, 7))
    plt.bar(plot_df["experiment_name"], plot_df["rmse"], color="#3a86ff")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Validation RMSE")
    plt.title("Tree Revalidation Experiments")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=180)
    plt.close()


def write_summary(results_df: pd.DataFrame, segment_df: pd.DataFrame, elapsed_sec: float) -> None:
    lines = [
        "# Tree Revalidation",
        "",
        "## Setup",
        "- Full-train reruns to test sampling bias",
        "- One-hot reruns to test categorical encoding impact",
        "- Tuned XGB/LGBM with early stopping",
        "- Metrics include RMSE/MAE/R2/RMSLE and segment metrics",
        "",
        "## Best Validation Runs",
    ]
    best_val = results_df.loc[results_df["split"] == "validation"].sort_values("rmse").head(12)
    for _, row in best_val.iterrows():
        lines.append(
            f"- {row['experiment_name']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, RMSLE={row['rmsle']:.4f}, R2={row['r2']:.4f}"
        )
    lines += ["", "## Best Test_2024 Runs"]
    best_test = results_df.loc[results_df["split"] == "test_2024"].sort_values("rmse").head(12)
    for _, row in best_test.iterrows():
        lines.append(
            f"- {row['experiment_name']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, RMSLE={row['rmsle']:.4f}, R2={row['r2']:.4f}"
        )
    lines += ["", f"Elapsed seconds: {elapsed_sec:.2f}"]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()
    train_df = df.loc[df["split"] == "train"].copy()
    validation_df = df.loc[df["split"] == "validation"].copy()
    test_dfs = {
        "validation": validation_df,
        "test_2023": df.loc[df["split"] == "test_2023"].copy(),
        "test_2024": df.loc[df["split"] == "test_2024"].copy(),
    }

    result_rows: list[dict] = []
    segment_rows: list[dict] = []

    for target_name in TARGETS:
        for experiment_name, (model, preprocessor, use_es) in make_fulltrain_models(target_name).items():
            metrics, segments = fit_predict_tree(train_df, validation_df, test_dfs, target_name, experiment_name, model, preprocessor, use_es)
            result_rows.extend(metrics)
            segment_rows.extend(segments)

        for experiment_name, (model, preprocessor, use_es) in make_encoding_models(target_name).items():
            metrics, segments = fit_predict_tree(train_df, validation_df, test_dfs, target_name, experiment_name, model, preprocessor, use_es)
            result_rows.extend(metrics)
            segment_rows.extend(segments)

        for experiment_name, (model, preprocessor, use_es) in make_tuned_models(target_name).items():
            metrics, segments = fit_predict_tree(train_df, validation_df, test_dfs, target_name, experiment_name, model, preprocessor, use_es)
            result_rows.extend(metrics)
            segment_rows.extend(segments)

    results_df = pd.DataFrame(result_rows)
    segment_df = pd.DataFrame(segment_rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    segment_df.to_csv(SEGMENT_CSV, index=False)
    save_plot(results_df)

    elapsed = time.time() - start
    summary = {
        "results_path": str(RESULTS_CSV),
        "segment_path": str(SEGMENT_CSV),
        "plot_path": str(PLOT_PATH),
        "elapsed_seconds": elapsed,
        "best_validation_rows": results_df.loc[results_df["split"] == "validation"].sort_values("rmse").head(10).to_dict(orient="records"),
        "best_test_2024_rows": results_df.loc[results_df["split"] == "test_2024"].sort_values("rmse").head(10).to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_summary(results_df, segment_df, elapsed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

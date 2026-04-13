from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import joblib
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


FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
INPUT_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "linear_baseline"
MODEL_PATH = OUTPUT_DIR / "linear_regression_pipeline.joblib"
METRICS_JSON = OUTPUT_DIR / "metrics.json"
METRICS_CSV = OUTPUT_DIR / "metrics.csv"
PRED_VAL_CSV = OUTPUT_DIR / "predictions_validation_2022.csv"
PRED_TEST_2023_CSV = OUTPUT_DIR / "predictions_test_2023.csv"
PRED_TEST_2024_CSV = OUTPUT_DIR / "predictions_test_2024.csv"
IMPORTANCE_CSV = OUTPUT_DIR / "feature_importance.csv"
IMPORTANCE_PNG = OUTPUT_DIR / "feature_importance_top20.png"
SUMMARY_MD = OUTPUT_DIR / "model_summary.md"

CATEGORICAL_FEATURES = [
    "sctg2",
    "dist_band",
    "trade_type",
    "orig_state_fips",
    "dest_state_fips",
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
]

USECOLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + [
    "target_tons",
    "target_log_tons",
    "split",
    "year",
    "dms_orig",
    "dms_dest",
]


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
        "dms_orig": "string",
        "dms_dest": "string",
    }
    df = pd.read_csv(INPUT_PATH, usecols=USECOLS, dtype=dtype_map, low_memory=False)
    return df


def build_pipeline() -> Pipeline:
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
    model = LinearRegression()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_split(
    df: pd.DataFrame,
    split_name: str,
    pipeline: Pipeline,
    pred_log_min: float,
    pred_log_max: float,
) -> tuple[dict, pd.DataFrame]:
    split_df = df.loc[df["split"] == split_name].copy()
    X = split_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_true = split_df["target_tons"].to_numpy()
    y_pred_tons = pipeline.predict(X)
    y_pred_tons = np.clip(y_pred_tons, a_min=0, a_max=None)

    metrics = {
        "split": split_name,
        "year_min": int(split_df["year"].min()),
        "year_max": int(split_df["year"].max()),
        "row_count": int(split_df.shape[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred_tons))),
        "mae": float(mean_absolute_error(y_true, y_pred_tons)),
        "r2": float(r2_score(y_true, y_pred_tons)),
    }

    prediction_df = split_df[["year", "dms_orig", "dms_dest", "target_tons"]].copy()
    prediction_df["predicted_tons"] = y_pred_tons
    prediction_df["abs_error"] = np.abs(prediction_df["target_tons"] - prediction_df["predicted_tons"])
    prediction_df["signed_error"] = prediction_df["predicted_tons"] - prediction_df["target_tons"]
    return metrics, prediction_df


def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefs = model.coef_
    importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    ).sort_values("abs_coefficient", ascending=False)
    return importance


def save_importance_plot(importance: pd.DataFrame) -> None:
    top20 = importance.head(20).iloc[::-1]
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 8))
    plt.barh(top20["feature_name"], top20["abs_coefficient"], color="#1f77b4")
    plt.xlabel("Absolute Coefficient")
    plt.ylabel("Feature")
    plt.title("Linear Regression Feature Importance (Top 20)")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PNG, dpi=180)
    plt.close()


def write_summary(metrics_df: pd.DataFrame, importance: pd.DataFrame) -> None:
    lines = [
        "# Linear Regression Baseline",
        "",
        "## Setup",
        "- Target used for training: `target_tons`",
        "- Evaluation scale: `tons`",
        "- Split policy: train=2020-2021, validation=2022, test_2023=2023, test_2024=2024",
        "- Categorical encoding: one-hot",
        "- Numeric preprocessing: median imputation + standardization",
        "",
        "## Metrics",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['split']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}, rows={int(row['row_count']):,}"
        )
    lines += [
        "",
        "## Top Coefficients",
    ]
    for _, row in importance.head(10).iterrows():
        lines.append(f"- {row['feature_name']}: coef={row['coefficient']:.6f}")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()

    train_df = df.loc[df["split"] == "train"].copy()
    X_train = train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_train = train_df["target_tons"].to_numpy()
    pred_log_min = float(np.nanmin(y_train))
    pred_log_max = float(np.nanmax(y_train))

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)

    metric_rows = []
    split_to_path = {
        "validation": PRED_VAL_CSV,
        "test_2023": PRED_TEST_2023_CSV,
        "test_2024": PRED_TEST_2024_CSV,
    }
    for split_name, path in split_to_path.items():
        metrics, pred_df = evaluate_split(df, split_name, pipeline, pred_log_min, pred_log_max)
        metric_rows.append(metrics)
        pred_df.to_csv(path, index=False)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    metrics_payload = {
        "generated_at_epoch": int(time.time()),
        "elapsed_seconds": round(time.time() - start, 2),
        "target": "target_tons",
        "evaluation_scale": "tons",
        "prediction_clip_range": [pred_log_min, pred_log_max],
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "metrics": metrics_df.to_dict(orient="records"),
    }
    METRICS_JSON.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    importance = extract_feature_importance(pipeline)
    importance.to_csv(IMPORTANCE_CSV, index=False)
    save_importance_plot(importance)
    write_summary(metrics_df, importance)

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

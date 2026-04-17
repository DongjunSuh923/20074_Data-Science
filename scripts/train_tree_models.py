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
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
BASELINE_METRICS_PATH = PROJECT_ROOT / "outputs" / "models" / "baseline_suite" / "baseline_metrics.csv"
INPUT_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "tree_models"
METRICS_CSV = OUTPUT_DIR / "tree_model_metrics.csv"
COMPARISON_CSV = OUTPUT_DIR / "model_comparison_with_baselines.csv"
SUMMARY_JSON = OUTPUT_DIR / "tree_model_summary.json"
SUMMARY_MD = OUTPUT_DIR / "tree_model_summary.md"

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

USECOLS = CATEGORICAL_FEATURES + NUMERIC_FEATURES + ["target_tons", "split", "year"]
TRAIN_SAMPLE_CAP = 400_000
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


def make_preprocessor() -> ColumnTransformer:
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
        ],
        remainder="drop",
    )


def get_models() -> dict[str, object]:
    return {
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=18,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBRegressor(
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
        "lightgbm": LGBMRegressor(
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


def fit_model(name: str, model: object, df: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    train_df = df.loc[df["split"] == "train"].copy()
    if train_df.shape[0] > TRAIN_SAMPLE_CAP:
        train_df = train_df.sample(TRAIN_SAMPLE_CAP, random_state=RANDOM_STATE)

    preprocessor = make_preprocessor()
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

    metric_rows = []
    feature_names = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.zeros(len(feature_names))
    importance_df = pd.DataFrame(
        {"feature_name": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / f"{name}_feature_importance.csv", index=False)
    save_importance_plot(name, importance_df)

    for split_name in ["validation", "test_2023", "test_2024"]:
        split_df = df.loc[df["split"] == split_name].copy()
        X_split = preprocessor.transform(split_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
        pred = np.asarray(model.predict(X_split))
        pred = np.clip(pred, a_min=0, a_max=None)
        metric_rows.append(metric_dict(name, split_name, split_df, pred))
        pred_df = split_df[["year", "target_tons"]].copy()
        pred_df["predicted_tons"] = pred
        pred_df["abs_error"] = np.abs(pred_df["target_tons"] - pred_df["predicted_tons"])
        pred_df.to_csv(OUTPUT_DIR / f"{name}_{split_name}_predictions.csv", index=False)

    return metric_rows, importance_df


def save_importance_plot(model_name: str, importance_df: pd.DataFrame) -> None:
    top = importance_df.head(15).iloc[::-1]
    plt.style.use("ggplot")
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature_name"], top["importance"], color="#3a86ff")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"{model_name} Feature Importance (Top 15)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_feature_importance_top15.png", dpi=180)
    plt.close()


def write_summary(metrics_df: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    lines = [
        "# Tree Model Summary",
        "",
        "## Setup",
        f"- Train sample cap per tree model: {TRAIN_SAMPLE_CAP:,}",
        "- Validation/test were evaluated on full splits.",
        "- Tree models used the same confirmed feature set as the baseline suite.",
        "",
        "## Tree Model Metrics",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"- {row['model_name']} / {row['split']}: RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}"
        )
    lines += [
        "",
        "## Comparison With Baselines",
    ]
    for split_name in ["validation", "test_2023", "test_2024"]:
        subset = comparison_df.loc[comparison_df["split"] == split_name].sort_values("rmse")
        lines.append(f"- {split_name}: best RMSE model = {subset.iloc[0]['model_name']} ({subset.iloc[0]['rmse']:.4f})")
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()
    models = get_models()

    metric_rows = []
    for name, model in models.items():
        rows, _ = fit_model(name, model, df)
        metric_rows.extend(rows)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    baseline_df = pd.read_csv(BASELINE_METRICS_PATH)
    baseline_keep = baseline_df.loc[baseline_df["model_name"].isin(["persistence_tons_lag1", "linear_full"])].copy()
    comparison_df = pd.concat(
        [
            baseline_keep[["model_name", "split", "rmse", "mae", "r2"]],
            metrics_df[["model_name", "split", "rmse", "mae", "r2"]],
        ],
        ignore_index=True,
    )
    comparison_df.to_csv(COMPARISON_CSV, index=False)

    summary = {
        "generated_at_epoch": int(time.time()),
        "elapsed_seconds": round(time.time() - start, 2),
        "train_sample_cap": TRAIN_SAMPLE_CAP,
        "metrics": metrics_df.to_dict(orient="records"),
        "comparison": comparison_df.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary(metrics_df, comparison_df)
    print(json.dumps({"elapsed_seconds": summary["elapsed_seconds"], "metrics_csv": str(METRICS_CSV), "comparison_csv": str(COMPARISON_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

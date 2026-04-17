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


FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
INPUT_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "baseline_suite"
METRICS_CSV = OUTPUT_DIR / "baseline_metrics.csv"
ABLATION_CSV = OUTPUT_DIR / "feature_group_ablation.csv"
SUMMARY_JSON = OUTPUT_DIR / "baseline_suite_summary.json"
SUMMARY_MD = OUTPUT_DIR / "baseline_suite_summary.md"
DATA_BASELINE_MD = OUTPUT_DIR / "official_modeling_baseline.md"
ABLATION_PNG = OUTPUT_DIR / "feature_group_ablation_validation_rmse.png"

GROUP_FEATURES = {
    "history": {
        "categorical": [],
        "numeric": ["tons_lag1", "tons_lag2", "tons_growth_rate", "tons_zero_flag_lag1"],
    },
    "cargo_distance": {
        "categorical": ["sctg2", "dist_band", "trade_type"],
        "numeric": [],
    },
    "region": {
        "categorical": ["orig_state_fips", "dest_state_fips"],
        "numeric": [],
    },
    "value_efficiency": {
        "categorical": [],
        "numeric": ["value_per_ton_lag1"],
    },
    "time": {
        "categorical": [],
        "numeric": ["year_index"],
    },
    "network": {
        "categorical": [],
        "numeric": [
            "orig_total_outbound_tons_lag1",
            "dest_total_inbound_tons_lag1",
            "orig_unique_dest_count_lag1",
            "dest_unique_orig_count_lag1",
            "route_share_of_origin_lag1",
            "route_share_of_dest_lag1",
            "corridor_sctg2_count_lag1",
        ],
    },
}

ALL_CATEGORICAL = [feat for group in GROUP_FEATURES.values() for feat in group["categorical"]]
ALL_NUMERIC = [feat for group in GROUP_FEATURES.values() for feat in group["numeric"]]
USECOLS = ALL_CATEGORICAL + ALL_NUMERIC + ["target_tons", "split", "year", "dms_orig", "dms_dest"]


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
    return pd.read_csv(INPUT_PATH, usecols=USECOLS, dtype=dtype_map, low_memory=False)


def build_pipeline(categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    transformers = []
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=True)),
                    ]
                ),
                categorical_features,
            )
        )
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0.3)
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", LinearRegression())])


def metric_dict(split_name: str, year_min: int, year_max: int, row_count: int, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "split": split_name,
        "year_min": int(year_min),
        "year_max": int(year_max),
        "row_count": int(row_count),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_predictions(df: pd.DataFrame, split_name: str, y_pred: np.ndarray) -> dict:
    split_df = df.loc[df["split"] == split_name]
    y_true = split_df["target_tons"].to_numpy()
    return metric_dict(
        split_name,
        int(split_df["year"].min()),
        int(split_df["year"].max()),
        int(split_df.shape[0]),
        y_true,
        y_pred,
    )


def run_persistence_baseline(df: pd.DataFrame) -> list[dict]:
    results = []
    for split_name in ["validation", "test_2023", "test_2024"]:
        split_df = df.loc[df["split"] == split_name]
        pred = split_df["tons_lag1"].clip(lower=0).to_numpy()
        result = evaluate_predictions(df, split_name, pred)
        result["model_name"] = "persistence_tons_lag1"
        result["feature_group_variant"] = "history_only"
        results.append(result)
    return results


def fit_and_evaluate_linear(
    df: pd.DataFrame,
    model_name: str,
    categorical_features: list[str],
    numeric_features: list[str],
) -> list[dict]:
    train_df = df.loc[df["split"] == "train"].copy()
    X_train = train_df[categorical_features + numeric_features]
    y_train = train_df["target_tons"].to_numpy()

    pipeline = build_pipeline(categorical_features, numeric_features)
    pipeline.fit(X_train, y_train)

    rows = []
    for split_name in ["validation", "test_2023", "test_2024"]:
        split_df = df.loc[df["split"] == split_name].copy()
        pred = pipeline.predict(split_df[categorical_features + numeric_features])
        pred = np.clip(pred, a_min=0, a_max=None)
        result = evaluate_predictions(df, split_name, pred)
        result["model_name"] = model_name
        result["feature_group_variant"] = model_name
        rows.append(result)
    return rows


def run_group_ablation(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    full_rows = fit_and_evaluate_linear(df, "linear_full", ALL_CATEGORICAL, ALL_NUMERIC)
    ablation_rows = []
    for group_name in GROUP_FEATURES:
        keep_categorical = [feat for feat in ALL_CATEGORICAL if feat not in GROUP_FEATURES[group_name]["categorical"]]
        keep_numeric = [feat for feat in ALL_NUMERIC if feat not in GROUP_FEATURES[group_name]["numeric"]]
        rows = fit_and_evaluate_linear(df, f"linear_minus_{group_name}", keep_categorical, keep_numeric)
        for row in rows:
            row["ablated_group"] = group_name
        ablation_rows.extend(rows)
    return full_rows, ablation_rows


def write_official_baseline_doc() -> None:
    lines = [
        "# Official Modeling Baseline",
        "",
        "## Official Input Files",
        f"- Feature dataset with splits: `{INPUT_PATH}`",
        f"- Truck-only preprocessing base: `{FEATURE_DIR.parent / 'preprocessing_truck_only'}`",
        "",
        "## Official Split Policy",
        "- Train: 2020-2021",
        "- Validation: 2022",
        "- Test 1: 2023",
        "- Test 2: 2024",
        "",
        "## Official Geography Policy",
        "- Primary geography: `dms_orig`, `dms_dest`",
        "- `fr_orig`, `fr_dest` are foreign-region fields and are not the main spatial unit for domestic truck forecasting.",
        "",
        "## Baseline Comparison Policy",
        "- Baseline 1: Persistence (`next tons = tons_lag1`)",
        "- Baseline 2: Linear Regression",
        "- Any later ML model should be compared against both baselines.",
        "",
        "## Feature Validation Policy",
        "- Interpret feature effects by group, not by single coefficient alone.",
        "- Use feature-group ablation before adding external data.",
    ]
    DATA_BASELINE_MD.write_text("\n".join(lines), encoding="utf-8")


def plot_ablation(ablation_df: pd.DataFrame, full_validation_rmse: float) -> None:
    validation_df = ablation_df.loc[ablation_df["split"] == "validation"].copy()
    validation_df["rmse_delta_vs_full"] = validation_df["rmse"] - full_validation_rmse
    validation_df = validation_df.sort_values("rmse_delta_vs_full", ascending=False)

    plt.style.use("ggplot")
    plt.figure(figsize=(10, 6))
    plt.barh(validation_df["ablated_group"], validation_df["rmse_delta_vs_full"], color="#bc4749")
    plt.xlabel("Validation RMSE increase vs full model")
    plt.ylabel("Ablated feature group")
    plt.title("Feature Group Ablation Impact")
    plt.tight_layout()
    plt.savefig(ABLATION_PNG, dpi=180)
    plt.close()


def write_summary(metrics_df: pd.DataFrame, ablation_df: pd.DataFrame) -> None:
    full_val_rmse = float(metrics_df.loc[(metrics_df["model_name"] == "linear_full") & (metrics_df["split"] == "validation"), "rmse"].iloc[0])
    persistence_val_rmse = float(metrics_df.loc[(metrics_df["model_name"] == "persistence_tons_lag1") & (metrics_df["split"] == "validation"), "rmse"].iloc[0])
    validation_ablation = ablation_df.loc[ablation_df["split"] == "validation"].copy()
    validation_ablation["rmse_delta_vs_full"] = validation_ablation["rmse"] - full_val_rmse
    validation_ablation = validation_ablation.sort_values("rmse_delta_vs_full", ascending=False)

    lines = [
        "# Baseline Suite Summary",
        "",
        "## Baselines",
        f"- Persistence validation RMSE: {persistence_val_rmse:.4f}",
        f"- Linear full validation RMSE: {full_val_rmse:.4f}",
        "",
        "## Interpretation",
        "- Persistence is the minimum bar because this dataset is strongly autoregressive.",
        "- Feature groups should be judged by how much validation/test performance drops when the group is removed.",
        "",
        "## Validation Ablation Ranking",
    ]
    for _, row in validation_ablation.iterrows():
        lines.append(
            f"- remove `{row['ablated_group']}`: RMSE delta {row['rmse_delta_vs_full']:.4f} (validation RMSE {row['rmse']:.4f})"
        )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()
    write_official_baseline_doc()

    metric_rows = []
    metric_rows.extend(run_persistence_baseline(df))
    full_rows, ablation_rows = run_group_ablation(df)
    metric_rows.extend(full_rows)
    metric_rows.extend(ablation_rows)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)

    full_metrics = metrics_df.loc[metrics_df["model_name"] == "linear_full", ["split", "rmse", "mae", "r2"]].rename(
        columns={"rmse": "full_rmse", "mae": "full_mae", "r2": "full_r2"}
    )
    ablation_df = metrics_df.loc[metrics_df["model_name"].str.startswith("linear_minus_")].merge(full_metrics, on="split", how="left")
    ablation_df["rmse_delta_vs_full"] = ablation_df["rmse"] - ablation_df["full_rmse"]
    ablation_df["mae_delta_vs_full"] = ablation_df["mae"] - ablation_df["full_mae"]
    ablation_df["r2_delta_vs_full"] = ablation_df["r2"] - ablation_df["full_r2"]
    ablation_df.to_csv(ABLATION_CSV, index=False)

    full_validation_rmse = float(full_metrics.loc[full_metrics["split"] == "validation", "full_rmse"].iloc[0])
    plot_ablation(ablation_df, full_validation_rmse)
    write_summary(metrics_df, ablation_df)

    summary_payload = {
        "generated_at_epoch": int(time.time()),
        "elapsed_seconds": round(time.time() - start, 2),
        "official_input": str(INPUT_PATH),
        "metrics": metrics_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"elapsed_seconds": summary_payload["elapsed_seconds"], "metrics_csv": str(METRICS_CSV), "ablation_csv": str(ABLATION_CSV)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

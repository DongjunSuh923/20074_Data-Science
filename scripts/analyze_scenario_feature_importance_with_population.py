from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for dep_dir in [PROJECT_ROOT / ".pydeps_fix", PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from xgboost import XGBRegressor


INPUT_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features" / "scenario_dataset_with_population.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features" / "feature_importance"
PERMUTATION_CSV = OUTPUT_DIR / "scenario_permutation_importance_with_population.csv"
GROUP_CSV = OUTPUT_DIR / "scenario_group_importance_with_population.csv"
SUMMARY_MD = OUTPUT_DIR / "scenario_feature_importance_with_population_summary.md"
SUMMARY_JSON = OUTPUT_DIR / "scenario_feature_importance_with_population_summary.json"
XGB_PNG = OUTPUT_DIR / "xgboost_top15_with_population.png"
RF_PNG = OUTPUT_DIR / "random_forest_top15_with_population.png"

CATEGORICAL = ["orig_state_fips", "dest_state_fips", "sctg2", "dist_band", "trade_type"]
NUMERIC = [
    "year_index",
    "orig_gdp_total","dest_gdp_total","orig_gdp_total_growth","dest_gdp_total_growth",
    "orig_gdp_manufacturing","dest_gdp_manufacturing","orig_gdp_wholesale","dest_gdp_wholesale",
    "orig_gdp_transport_warehousing","dest_gdp_transport_warehousing",
    "orig_gdp_mfg_share","dest_gdp_mfg_share","orig_gdp_wholesale_share","dest_gdp_wholesale_share",
    "orig_gdp_transport_warehousing_share","dest_gdp_transport_warehousing_share",
    "orig_cbp_total_emp","dest_cbp_total_emp","orig_cbp_mfg_emp","dest_cbp_mfg_emp",
    "orig_cbp_wholesale_emp","dest_cbp_wholesale_emp","orig_cbp_transport_emp","dest_cbp_transport_emp",
    "orig_cbp_warehousing_emp","dest_cbp_warehousing_emp","orig_cbp_total_est","dest_cbp_total_est",
    "orig_cbp_warehousing_est","dest_cbp_warehousing_est","route_gdp_total_sum","route_gdp_total_gap",
    "route_cbp_total_emp_sum","route_cbp_warehousing_emp_sum","route_cbp_transport_emp_sum",
    "orig_population","dest_population","orig_population_growth","dest_population_growth",
    "orig_gdp_per_capita","dest_gdp_per_capita","route_population_sum","route_population_gap","route_gdp_per_capita_gap",
]

FEATURE_GROUP_MAP = {col: "internal_route_structure" for col in CATEGORICAL}
FEATURE_GROUP_MAP["year_index"] = "internal_time"
for col in NUMERIC:
    if col not in FEATURE_GROUP_MAP:
        if "population" in col or "per_capita" in col:
            FEATURE_GROUP_MAP[col] = "external_population"
        else:
            FEATURE_GROUP_MAP[col] = "external_state_economy"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, dtype={col: "string" for col in CATEGORICAL + ["split"]}, low_memory=False)


def build_xgb_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ]
    )
    model = XGBRegressor(
        n_estimators=500,max_depth=8,learning_rate=0.05,subsample=0.85,colsample_bytree=0.85,
        reg_lambda=1.0,objective="reg:squarederror",tree_method="hist",random_state=42,n_jobs=-1
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def build_rf_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), CATEGORICAL),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC),
        ],
        sparse_threshold=0.2,
    )
    model = RandomForestRegressor(n_estimators=200, max_depth=16, min_samples_leaf=2, n_jobs=-1, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def run_permutation(model_name: str, pipeline: Pipeline, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    X_train = train_df[CATEGORICAL + NUMERIC]
    y_train = train_df["target_tons"].to_numpy()
    pipeline.fit(X_train, y_train)
    sample_df = val_df.sample(min(30000, len(val_df)), random_state=42).copy()
    X_val = sample_df[CATEGORICAL + NUMERIC]
    y_val = sample_df["target_tons"].to_numpy()
    result = permutation_importance(
        pipeline, X_val, y_val, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=42, n_jobs=1
    )
    out = pd.DataFrame(
        {"model_name": model_name, "feature_name": CATEGORICAL + NUMERIC, "importance_mean": result.importances_mean, "importance_std": result.importances_std}
    )
    out["feature_group"] = out["feature_name"].map(FEATURE_GROUP_MAP)
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def save_plot(df: pd.DataFrame, title: str, path: Path) -> None:
    top = df.head(15).iloc[::-1]
    plt.style.use("ggplot")
    plt.figure(figsize=(10, 7))
    plt.barh(top["feature_name"], top["importance_mean"], color="#3a86ff")
    plt.xlabel("Permutation Importance (RMSE drop)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    start = time.time()
    df = load_data()
    train_df = df.loc[df["split"] == "train"].copy()
    val_df = df.loc[df["split"] == "validation"].copy()

    xgb_df = run_permutation("xgboost", build_xgb_pipeline(), train_df, val_df)
    rf_df = run_permutation("random_forest", build_rf_pipeline(), train_df, val_df)
    permutation_df = pd.concat([xgb_df, rf_df], ignore_index=True)
    permutation_df.to_csv(PERMUTATION_CSV, index=False)

    group_df = (
        permutation_df.groupby(["model_name", "feature_group"], as_index=False)["importance_mean"]
        .sum()
        .rename(columns={"importance_mean": "importance_sum"})
    )
    group_df.to_csv(GROUP_CSV, index=False)

    save_plot(xgb_df, "XGBoost Top 15 With Population", XGB_PNG)
    save_plot(rf_df, "Random Forest Top 15 With Population", RF_PNG)

    summary = {
        "permutation_path": str(PERMUTATION_CSV),
        "group_path": str(GROUP_CSV),
        "elapsed_seconds": time.time() - start,
        "top_xgboost_features": xgb_df.head(10).to_dict(orient="records"),
        "top_random_forest_features": rf_df.head(10).to_dict(orient="records"),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Scenario Feature Importance With Population", "", "## Top XGBoost Features"]
    for _, row in xgb_df.head(10).iterrows():
        lines.append(f"- {row['feature_name']} ({row['feature_group']}): {row['importance_mean']:.4f}")
    lines += ["", "## Top Random Forest Features"]
    for _, row in rf_df.head(10).iterrows():
        lines.append(f"- {row['feature_name']} ({row['feature_group']}): {row['importance_mean']:.4f}")
    lines += ["", "## Group Importance"]
    for _, row in group_df.sort_values(["model_name", "importance_sum"], ascending=[True, False]).iterrows():
        lines.append(f"- {row['model_name']} / {row['feature_group']}: {row['importance_sum']:.4f}")
    lines += ["", f"Elapsed seconds: {time.time() - start:.2f}"]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

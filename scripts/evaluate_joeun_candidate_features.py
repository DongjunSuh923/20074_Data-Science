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
JOEUN_DIR = PROJECT_ROOT / "outputs" / "external_features" / "joeun" / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "joeun_feature_eval"
DATASET_PATH = OUTPUT_DIR / "scenario_dataset_with_joeun_candidates.csv"
RESULTS_PATH = OUTPUT_DIR / "joeun_candidate_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "joeun_candidate_summary.md"
SUMMARY_JSON = OUTPUT_DIR / "joeun_candidate_summary.json"

CATEGORICAL = ["orig_state_fips", "dest_state_fips", "sctg2", "dist_band", "trade_type"]
BASE_NUMERIC = [
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
]

FEATURE_SETS = {
    "baseline_rf_population": BASE_NUMERIC,
    "plus_real_mfg_gdp": BASE_NUMERIC + [
        "orig_real_mfg_gdp",
        "dest_real_mfg_gdp",
        "route_real_mfg_gdp_gap",
        "route_real_mfg_gdp_sum",
    ],
    "plus_accidents": BASE_NUMERIC + [
        "orig_highway_accidents",
        "dest_highway_accidents",
        "route_highway_accident_gap",
        "route_highway_accident_sum",
    ],
    "plus_both": BASE_NUMERIC + [
        "orig_real_mfg_gdp",
        "dest_real_mfg_gdp",
        "route_real_mfg_gdp_gap",
        "route_real_mfg_gdp_sum",
        "orig_highway_accidents",
        "dest_highway_accidents",
        "route_highway_accident_gap",
        "route_highway_accident_sum",
    ],
}

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10",
    "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19",
    "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27",
    "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35",
    "NY": "36", "NC": "37", "ND": "38", "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53",
    "WV": "54", "WI": "55", "WY": "56",
}


def load_base() -> pd.DataFrame:
    return pd.read_csv(INPUT_PATH, dtype={col: "string" for col in CATEGORICAL + ["split"]}, low_memory=False)


def load_real_mfg() -> pd.DataFrame:
    df = pd.read_csv(JOEUN_DIR / "state_manufacturing_gdp_2018_2024.csv")
    df["year"] = pd.to_datetime(df["observation_date"]).dt.year.astype(int)
    df["state_fips"] = df["State"].map(STATE_ABBR_TO_FIPS)
    df = df.rename(columns={"Real_Manufacturing_GDP": "real_mfg_gdp"})
    return df[["state_fips", "year", "real_mfg_gdp"]].dropna()


def load_accidents() -> pd.DataFrame:
    df = pd.read_csv(JOEUN_DIR / "state_highway_accidents_count_2018_2024.csv")
    df["state_fips"] = df["State"].map(STATE_ABBR_TO_FIPS)
    df = df.rename(columns={"Year": "year", "Highway_Accident_Count": "highway_accidents"})
    return df[["state_fips", "year", "highway_accidents"]].dropna()


def attach_joeun_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    real_mfg = load_real_mfg()
    accidents = load_accidents()

    out = out.merge(
        real_mfg.rename(columns={"state_fips": "orig_state_fips", "real_mfg_gdp": "orig_real_mfg_gdp"}),
        on=["orig_state_fips", "year"],
        how="left",
    )
    out = out.merge(
        real_mfg.rename(columns={"state_fips": "dest_state_fips", "real_mfg_gdp": "dest_real_mfg_gdp"}),
        on=["dest_state_fips", "year"],
        how="left",
    )
    out["route_real_mfg_gdp_gap"] = (out["orig_real_mfg_gdp"] - out["dest_real_mfg_gdp"]).abs()
    out["route_real_mfg_gdp_sum"] = out["orig_real_mfg_gdp"] + out["dest_real_mfg_gdp"]

    out = out.merge(
        accidents.rename(columns={"state_fips": "orig_state_fips", "highway_accidents": "orig_highway_accidents"}),
        on=["orig_state_fips", "year"],
        how="left",
    )
    out = out.merge(
        accidents.rename(columns={"state_fips": "dest_state_fips", "highway_accidents": "dest_highway_accidents"}),
        on=["dest_state_fips", "year"],
        how="left",
    )
    out["route_highway_accident_gap"] = (out["orig_highway_accidents"] - out["dest_highway_accidents"]).abs()
    out["route_highway_accident_sum"] = out["orig_highway_accidents"] + out["dest_highway_accidents"]

    return out


def build_rf(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                ]),
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
    return Pipeline([("preprocessor", pre), ("model", model)])


def build_xgb(numeric_features: list[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
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
    return Pipeline([("preprocessor", pre), ("model", model)])


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

    df = attach_joeun_features(load_base())
    df.to_csv(DATASET_PATH, index=False)

    train_df = df.loc[df["split"] == "train"].copy()
    y_train = train_df["target_tons"].to_numpy()

    rows: list[dict] = []
    for feature_set_name, numeric_features in FEATURE_SETS.items():
        feature_cols = CATEGORICAL + numeric_features
        X_train = train_df[feature_cols]

        for model_name, builder in {"xgboost": build_xgb, "random_forest": build_rf}.items():
            pipe = builder(numeric_features)
            pipe.fit(X_train, y_train)
            for split_name in ["validation", "test_2023", "test_2024"]:
                split_df = df.loc[df["split"] == split_name].copy()
                pred = np.clip(pipe.predict(split_df[feature_cols]), a_min=0, a_max=None)
                rows.append(metric_row(model_name, feature_set_name, split_name, split_df["target_tons"].to_numpy(), pred))

    results = pd.DataFrame(rows).sort_values(["split", "rmse"])
    results.to_csv(RESULTS_PATH, index=False)

    best_by_split = results.groupby("split", as_index=False).first()
    lines = [
        "# Joeun Candidate Feature Evaluation",
        "",
        "- Base set: rf_population compact set",
        "- Added candidates: real manufacturing GDP, highway accident count",
        "",
        "## Best By Split",
    ]
    for _, row in best_by_split.iterrows():
        lines.append(
            f"- {row['split']}: {row['model_name']} / {row['feature_set']} "
            f"(RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}, R2={row['r2']:.4f}, features={int(row['feature_count'])})"
        )
    lines += ["", f"Elapsed seconds: {time.time() - start:.2f}"]
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "dataset_path": str(DATASET_PATH),
        "results_path": str(RESULTS_PATH),
        "best_by_split": best_by_split.to_dict(orient="records"),
        "elapsed_seconds": time.time() - start,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

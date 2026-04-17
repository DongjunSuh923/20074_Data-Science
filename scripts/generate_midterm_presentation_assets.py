from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for dep_dir in [PROJECT_ROOT / ".pydeps_fix", PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = PROJECT_ROOT / "outputs" / "presentation_midterm"
SUMMARY_JSON = OUTPUT_DIR / "presentation_asset_summary.json"
SUMMARY_MD = OUTPUT_DIR / "presentation_asset_summary.md"
METRICS_CSV = OUTPUT_DIR / "presentation_key_metrics.csv"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_yearly_total_tons() -> Path:
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "eda" / "yearly_totals.csv")
    fig_path = OUTPUT_DIR / "01_yearly_total_tons.png"
    plt.figure(figsize=(8, 4.5))
    plt.plot(df["year"], df["total_tons"], marker="o", linewidth=2.2, color="#1f77b4")
    plt.title("Truck-Only FAF5: Total Tons by Year")
    plt.xlabel("Year")
    plt.ylabel("Total Tons")
    plt.grid(alpha=0.25)
    savefig(fig_path)
    return fig_path


def plot_lag_dominance() -> Path:
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "models" / "lag_role_experiments" / "lag_role_results.csv")
    keep = ["persistence", "lag_only_linear", "full_linear", "no_lag_linear", "delta_nonlag_linear", "delta_with_lag_linear"]
    df = df[(df["split"] == "validation") & (df["experiment_name"].isin(keep))].copy()
    order = ["persistence", "lag_only_linear", "full_linear", "delta_with_lag_linear", "delta_nonlag_linear", "no_lag_linear"]
    label_map = {
        "persistence": "Persistence",
        "lag_only_linear": "Lag Only",
        "full_linear": "Full Linear",
        "delta_with_lag_linear": "Delta + Lag",
        "delta_nonlag_linear": "Delta No-Lag",
        "no_lag_linear": "No-Lag Linear",
    }
    df["experiment_name"] = pd.Categorical(df["experiment_name"], categories=order, ordered=True)
    df = df.sort_values("experiment_name")
    fig_path = OUTPUT_DIR / "02_lag_dominance_validation_rmse.png"
    plt.figure(figsize=(9.2, 4.8))
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#eeca3b", "#f58518", "#e45756"]
    plt.bar([label_map[x] for x in df["experiment_name"]], df["rmse"], color=colors[: len(df)])
    plt.title("Validation RMSE: Lag Dominance Check")
    plt.ylabel("RMSE")
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig(fig_path)
    return fig_path


def plot_scenario_model_comparison() -> Path:
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_model_results.csv")
    df = df[df["target_name"] == "raw"].copy()
    pivot = df.pivot(index="model_name", columns="split", values="rmse").loc[
        ["linear", "random_forest", "xgboost", "lightgbm"]
    ]
    fig_path = OUTPUT_DIR / "03_lag_free_model_comparison.png"
    ax = pivot.plot(kind="bar", figsize=(9.5, 5.2), color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Lag-Free Scenario Models: RMSE Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Split")
    savefig(fig_path)
    return fig_path


def plot_population_impact() -> Path:
    base = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_model_results.csv")
    pop = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "population_features" / "scenario_model_results_with_population.csv")
    base = base[(base["target_name"] == "raw") & (base["model_name"].isin(["random_forest", "xgboost"]))].copy()
    pop = pop[(pop["target_name"] == "raw") & (pop["model_name"].isin(["random_forest", "xgboost"]))].copy()
    merged = base.merge(pop, on=["model_name", "split", "target_name"], suffixes=("_base", "_pop"))
    merged["rmse_delta"] = merged["rmse_pop"] - merged["rmse_base"]
    pivot = merged.pivot(index="model_name", columns="split", values="rmse_delta").loc[["random_forest", "xgboost"]]
    fig_path = OUTPUT_DIR / "04_population_feature_rmse_delta.png"
    ax = pivot.plot(kind="bar", figsize=(8.8, 4.8), color=["#4c78a8", "#f58518", "#54a24b"])
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Population/GDP per Capita Impact on RMSE")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE Delta (Population - Base)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Split")
    savefig(fig_path)
    return fig_path


def plot_feature_set_selection() -> Path:
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "feature_set_selection" / "feature_set_results.csv")
    df = df[df["split"].isin(["validation", "test_2023", "test_2024"])].copy()
    # Keep best of each feature_set across models by average RMSE
    avg = df.groupby(["feature_set", "model_name"], as_index=False)["rmse"].mean()
    avg["label"] = avg["model_name"] + "\n" + avg["feature_set"]
    avg = avg.sort_values("rmse")
    fig_path = OUTPUT_DIR / "05_feature_set_avg_rmse.png"
    plt.figure(figsize=(10.2, 5.4))
    plt.bar(avg["label"], avg["rmse"], color="#4c78a8")
    plt.title("Average RMSE by Compact Feature Set Candidate")
    plt.ylabel("Average RMSE across validation/test")
    plt.xticks(rotation=28, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig(fig_path)
    return fig_path


def plot_joeun_candidate_comparison() -> Path:
    df = pd.read_csv(PROJECT_ROOT / "outputs" / "scenario_model" / "joeun_feature_eval" / "joeun_candidate_results.csv")
    keep_sets = ["baseline_rf_population", "plus_real_mfg_gdp", "plus_accidents", "plus_both"]
    df = df[df["feature_set"].isin(keep_sets)].copy()
    avg = df.groupby(["feature_set", "model_name"], as_index=False)["rmse"].mean()
    avg["label"] = avg["model_name"] + "\n" + avg["feature_set"]
    avg = avg.sort_values("rmse")
    fig_path = OUTPUT_DIR / "06_joeun_candidate_avg_rmse.png"
    plt.figure(figsize=(10.0, 5.4))
    plt.bar(avg["label"], avg["rmse"], color="#54a24b")
    plt.title("Average RMSE: Joeun External Feature Candidates")
    plt.ylabel("Average RMSE across validation/test")
    plt.xticks(rotation=28, ha="right")
    plt.grid(axis="y", alpha=0.25)
    savefig(fig_path)
    return fig_path


def plot_final_candidate_comparison() -> Path:
    rows = [
        {"candidate": "RF + plus_both", "split": "validation", "rmse": 484.72124436042037},
        {"candidate": "RF + plus_both", "split": "test_2023", "rmse": 499.84769107452524},
        {"candidate": "RF + plus_both", "split": "test_2024", "rmse": 629.5698729424729},
        {"candidate": "XGB + plus_accidents", "split": "validation", "rmse": 426.10256330466575},
        {"candidate": "XGB + plus_accidents", "split": "test_2023", "rmse": 454.5093361083875},
        {"candidate": "XGB + plus_accidents", "split": "test_2024", "rmse": 690.8291897622305},
    ]
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="candidate", columns="split", values="rmse").loc[
        ["RF + plus_both", "XGB + plus_accidents"]
    ]
    fig_path = OUTPUT_DIR / "07_final_candidate_comparison.png"
    ax = pivot.plot(kind="bar", figsize=(8.4, 4.8), color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_title("Current Final Candidate Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("RMSE")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Split")
    savefig(fig_path)
    return fig_path


def build_metrics_table() -> Path:
    rows = [
        {"stage": "Initial lag-free model", "candidate": "XGBoost(raw)", "split": "validation", "rmse": 444.0640, "r2": 0.8837},
        {"stage": "Initial lag-free model", "candidate": "XGBoost(raw)", "split": "test_2023", "rmse": 540.3650, "r2": 0.8308},
        {"stage": "Initial lag-free model", "candidate": "RandomForest(raw)", "split": "test_2024", "rmse": 835.4419, "r2": 0.6008},
        {"stage": "Compact feature selection", "candidate": "XGBoost + rf_population", "split": "test_2023", "rmse": 509.1484, "r2": 0.8498},
        {"stage": "Compact feature selection", "candidate": "RandomForest + rf_population", "split": "test_2024", "rmse": 676.5496, "r2": 0.7382},
        {"stage": "Joeun candidate eval", "candidate": "XGBoost + plus_accidents", "split": "validation", "rmse": 426.1026, "r2": 0.8929},
        {"stage": "Joeun candidate eval", "candidate": "XGBoost + plus_accidents", "split": "test_2023", "rmse": 454.5093, "r2": 0.8803},
        {"stage": "Joeun candidate eval", "candidate": "RandomForest + plus_both", "split": "test_2024", "rmse": 629.5699, "r2": 0.7733},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_CSV, index=False)
    return METRICS_CSV


def main() -> None:
    ensure_output_dir()
    assets = {
        "yearly_total_tons": str(plot_yearly_total_tons()),
        "lag_dominance_validation": str(plot_lag_dominance()),
        "lag_free_model_comparison": str(plot_scenario_model_comparison()),
        "population_impact": str(plot_population_impact()),
        "feature_set_selection": str(plot_feature_set_selection()),
        "joeun_candidate_comparison": str(plot_joeun_candidate_comparison()),
        "final_candidate_comparison": str(plot_final_candidate_comparison()),
        "key_metrics_csv": str(build_metrics_table()),
    }

    lines = [
        "# Midterm Presentation Assets",
        "",
        "## Generated Assets",
    ]
    for key, value in assets.items():
        lines.append(f"- {key}: {value}")
    lines += [
        "",
        "## Suggested Use",
        "- 01: preprocessing and EDA trend slide",
        "- 02: lag dominance problem slide",
        "- 03: lag-free official model comparison slide",
        "- 04: population/per-capita GDP experiment slide",
        "- 05: compact feature set selection slide",
        "- 06: teammate external data evaluation slide",
        "- 07: final midterm model candidate comparison slide",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    SUMMARY_JSON.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()

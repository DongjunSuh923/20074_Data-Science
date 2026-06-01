from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
FHWA_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "fhwa_highway_spending_feature_prep" / "state_fhwa_highway_spending_features_2018_2024.csv"
QCEW_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "qcew_gravel_cereal_demand_prep" / "state_qcew_gravel_cereal_demand_features_2018_2024.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "construction_infra_shock"

SEVERITY_BY_RANK = {1: "severe", 2: "medium", 3: "medium", 4: "mild", 5: "mild"}
STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "DC": "11", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17",
    "IN": "18", "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24",
    "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31",
    "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46",
    "TN": "47", "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54",
    "WI": "55", "WY": "56",
}


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fhwa = pd.read_csv(FHWA_PATH)
    qcew = pd.read_csv(QCEW_PATH)

    fhwa_2024 = fhwa.loc[fhwa["year"] == 2024, ["state_fips", "fhwa_sf4_capital_outlay_roads_bridges_kusd", "fhwa_sf4_total_disbursements_kusd", "fhwa_sf4_reserves_current_highway_work_kusd"]].copy()
    fhwa_2024["state_fips"] = fhwa_2024["state_fips"].astype(str).str.zfill(2)
    qcew_2024 = qcew.loc[qcew["year"] == 2024, ["state_abbr", "ready_mix_concrete_emplvl", "cement_manufacturing_emplvl", "highway_bridge_construction_emplvl"]].copy()
    qcew_2024["state_fips"] = qcew_2024["state_abbr"].map(STATE_ABBR_TO_FIPS)

    merged = fhwa_2024.merge(qcew_2024, on="state_fips", how="left")
    numeric_cols = [
        "fhwa_sf4_capital_outlay_roads_bridges_kusd",
        "fhwa_sf4_total_disbursements_kusd",
        "fhwa_sf4_reserves_current_highway_work_kusd",
        "ready_mix_concrete_emplvl",
        "cement_manufacturing_emplvl",
        "highway_bridge_construction_emplvl",
    ]
    for col in numeric_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["construction_infra_score"] = (
        1.5 * zscore(np.log1p(merged["fhwa_sf4_capital_outlay_roads_bridges_kusd"]))
        + 1.0 * zscore(np.log1p(merged["fhwa_sf4_total_disbursements_kusd"]))
        + 0.5 * zscore(np.log1p(merged["ready_mix_concrete_emplvl"]))
        + 0.5 * zscore(np.log1p(merged["cement_manufacturing_emplvl"]))
        + 0.5 * zscore(np.log1p(merged["highway_bridge_construction_emplvl"]))
    )
    merged = merged.sort_values("construction_infra_score", ascending=False).reset_index(drop=True)
    merged["rank_within_scenario"] = np.arange(1, len(merged) + 1)
    merged["severity"] = merged["rank_within_scenario"].map(SEVERITY_BY_RANK)

    top = merged.loc[merged["rank_within_scenario"] <= 5].copy()
    top["scenario_name"] = "construction_infra_state_blocking"
    top["shock_family"] = "Demand shock"
    top["commodity_filter"] = "11,12,31"
    top.to_csv(OUT_DIR / "construction_infra_state_blocking_cases.csv", index=False)
    merged.to_csv(OUT_DIR / "construction_infra_state_scores.csv", index=False)

    lines = ["# Construction / Infrastructure Shock Inputs", ""]
    for row in top.itertuples(index=False):
        lines.append(
            f"- {row.state_fips}: severity={row.severity}, score={row.construction_infra_score:.3f}, "
            f"capital_outlay={row.fhwa_sf4_capital_outlay_roads_bridges_kusd:,.0f}, "
            f"disbursements={row.fhwa_sf4_total_disbursements_kusd:,.0f}, "
            f"ready_mix_empl={row.ready_mix_concrete_emplvl:,.0f}"
        )
    (OUT_DIR / "construction_infra_state_blocking_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

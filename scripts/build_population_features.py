from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POP_XLSX = PROJECT_ROOT / "outputs" / "external_features" / "nst-est2020int-pop.xlsx"
POP_CSV = PROJECT_ROOT / "outputs" / "external_features" / "NST-EST2025-POPCHG2020-2025.csv"
JOEUN_POP_CSV = PROJECT_ROOT / "outputs" / "external_features" / "joeun" / "data" / "state_population_2018_2024.csv"
SCENARIO_INPUT = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_dataset_state_to_state.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "population_features"

POP_OUTPUT = OUTPUT_DIR / "state_population_2018_2024.csv"
SCENARIO_OUTPUT = OUTPUT_DIR / "scenario_dataset_with_population.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "population_feature_summary.json"


def load_intercensal() -> pd.DataFrame:
    df = pd.read_excel(POP_XLSX, sheet_name="NST-EST2020INT-POP", header=3)
    df = df.rename(columns={df.columns[0]: "NAME"})
    df = df.loc[df["NAME"].notna()].copy()
    df = df.loc[df["NAME"].astype(str).str.strip() != ""].copy()
    # state rows only
    state_names = {
        "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware","District of Columbia",
        "Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana","Maine","Maryland",
        "Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska","Nevada","New Hampshire",
        "New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania",
        "Rhode Island","South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
        "West Virginia","Wisconsin","Wyoming",
    }
    df["NAME"] = df["NAME"].astype(str).str.strip().replace({"District of Columbia": "Washington DC"})
    df = df.loc[df["NAME"].isin(state_names | {"Washington DC"})].copy()

    year_map = {}
    for col in df.columns:
        if isinstance(col, (int, float)) and int(col) in {2018, 2019, 2020}:
            year_map[col] = int(col)
        elif isinstance(col, str) and col.strip().isdigit() and int(col.strip()) in {2018, 2019, 2020}:
            year_map[col] = int(col.strip())
    keep_cols = ["NAME"] + list(year_map.keys())
    df = df[keep_cols].rename(columns=year_map)
    long_df = df.melt(id_vars="NAME", var_name="year", value_name="population")
    long_df["year"] = long_df["year"].astype(int)
    long_df["population"] = pd.to_numeric(long_df["population"], errors="coerce")
    return long_df


def load_post2020() -> pd.DataFrame:
    df = pd.read_csv(POP_CSV)
    df = df.loc[df["SUMLEV"] == 40].copy()
    df["NAME"] = df["NAME"].astype(str).str.strip().replace({"District of Columbia": "Washington DC"})
    long_rows = []
    for year in [2020, 2021, 2022, 2023, 2024]:
        col = f"POPESTIMATE{year}"
        if col not in df.columns:
            continue
        tmp = df[["NAME", "STATE", col]].copy()
        tmp["year"] = year
        tmp = tmp.rename(columns={col: "population"})
        long_rows.append(tmp[["NAME", "year", "population"]])
    out = pd.concat(long_rows, ignore_index=True)
    out["population"] = pd.to_numeric(out["population"], errors="coerce")
    return out


def build_population_table() -> pd.DataFrame:
    if JOEUN_POP_CSV.exists():
        fred_pop = pd.read_csv(JOEUN_POP_CSV)
        fred_pop["year"] = pd.to_datetime(fred_pop["observation_date"]).dt.year
        fred_pop = fred_pop.rename(columns={"Population": "population", "State": "state_abbr"})
        state_lookup = pd.read_csv(
            PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "lookups" / "state_lookup.csv",
            dtype={"state_fips": "string"},
        )
        state_lookup["state_fips"] = state_lookup["state_fips"].str.zfill(2)
        abbr_lookup = pd.DataFrame(
            {
                "state_abbr": [
                    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME",
                    "MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI",
                    "SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
                ],
                "state_fips": [
                    "01","02","04","05","06","08","09","10","11","12","13","15","16","17","18","19","20","21","22","23",
                    "24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","44",
                    "45","46","47","48","49","50","51","53","54","55","56",
                ],
            }
        )
        pop = (
            fred_pop.merge(abbr_lookup, on="state_abbr", how="left")
            .merge(state_lookup, on="state_fips", how="left")
            [["state_fips", "state_name", "year", "population"]]
            .copy()
        )
        pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
        pop = pop.sort_values(["state_fips", "year"])
        pop["population_growth"] = pop.groupby("state_fips")["population"].pct_change()
        return pop

    a = load_intercensal()
    b = load_post2020()
    combined = pd.concat([a, b], ignore_index=True)
    combined = combined.drop_duplicates(subset=["NAME", "year"], keep="last").sort_values(["NAME", "year"])
    state_lookup = pd.read_csv(
        PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "lookups" / "state_lookup.csv",
        dtype={"state_fips": "string"},
    )
    state_lookup["state_fips"] = state_lookup["state_fips"].str.zfill(2)
    pop = combined.merge(state_lookup, left_on="NAME", right_on="state_name", how="left")
    pop = pop[["state_fips", "state_name", "year", "population"]].copy()
    pop["population_growth"] = pop.groupby("state_fips")["population"].pct_change()
    return pop


def attach_population_and_gdppc(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    orig = pop.add_prefix("orig_").rename(columns={"orig_state_fips": "orig_state_fips", "orig_year": "year"})
    dest = pop.add_prefix("dest_").rename(columns={"dest_state_fips": "dest_state_fips", "dest_year": "year"})
    merged = df.merge(orig, on=["orig_state_fips", "year"], how="left")
    merged = merged.merge(dest, on=["dest_state_fips", "year"], how="left")
    merged["orig_gdp_per_capita"] = merged["orig_gdp_total"] * 1_000_000 / merged["orig_population"]
    merged["dest_gdp_per_capita"] = merged["dest_gdp_total"] * 1_000_000 / merged["dest_population"]
    merged["route_population_sum"] = merged["orig_population"] + merged["dest_population"]
    merged["route_population_gap"] = (merged["orig_population"] - merged["dest_population"]).abs()
    merged["route_gdp_per_capita_gap"] = (merged["orig_gdp_per_capita"] - merged["dest_gdp_per_capita"]).abs()
    return merged


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pop = build_population_table()
    pop.to_csv(POP_OUTPUT, index=False)

    scenario_df = pd.read_csv(
        SCENARIO_INPUT,
        dtype={"orig_state_fips": "string", "dest_state_fips": "string"},
        low_memory=False,
    )
    scenario_df["orig_state_fips"] = scenario_df["orig_state_fips"].str.zfill(2)
    scenario_df["dest_state_fips"] = scenario_df["dest_state_fips"].str.zfill(2)
    scenario_with_pop = attach_population_and_gdppc(scenario_df, pop)
    scenario_with_pop.to_csv(SCENARIO_OUTPUT, index=False)

    summary = {
        "population_table": str(POP_OUTPUT),
        "scenario_output": str(SCENARIO_OUTPUT),
        "population_years": sorted(pop["year"].unique().tolist()),
        "states": int(pop["state_fips"].nunique()),
        "added_columns": [
            "orig_population",
            "dest_population",
            "orig_population_growth",
            "dest_population_growth",
            "orig_gdp_per_capita",
            "dest_gdp_per_capita",
            "route_population_sum",
            "route_population_gap",
            "route_gdp_per_capita_gap",
        ],
    }
    SUMMARY_OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

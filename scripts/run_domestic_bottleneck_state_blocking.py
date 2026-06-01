from __future__ import annotations

import json
import re
from html import unescape
from pathlib import Path

import pandas as pd


READ_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

BASE_ROUTE_PATH = READ_ROOT / "outputs" / "scenario_model" / "hub_forecast_5y_forward_scenarios" / "baseline_forecast_routes_2025_2031.csv"
MUST_HAVE_PATH = READ_ROOT / "outputs" / "scenario_model" / "state_blocking_resilience" / "provisional_must_have_hubs.csv"
EXEC_SUMMARY_PATH = WORK_ROOT / "external_data" / "Scenario_Tier1" / "Domestic_Bottleneck" / "fhwa_freight_mobility_exec_summary_2019.html"
APPENDIX_B_PATH = WORK_ROOT / "external_data" / "Scenario_Tier1" / "Domestic_Bottleneck" / "fhwa_freight_mobility_appendix_b_2019.html"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "domestic_bottleneck_shock"
START_YEAR = 2026
TOP_N = 15

SEVERITY_RULES = {
    "mild": {"start_loss": 0.05, "duration_years": 1},
    "medium": {"start_loss": 0.20, "duration_years": 3},
    "severe": {"start_loss": 0.40, "duration_years": 5},
}

STATE_ABBR_MAP = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}
STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
STATE_ABBR_TO_FIPS = {abbr: fips for fips, abbr in STATE_ABBR_MAP.items()}
EXCLUDED_STATES = {"AK", "HI", "DC"}


def schedule(severity: str) -> dict[int, float]:
    rule = SEVERITY_RULES[severity]
    loss = rule["start_loss"]
    out = {}
    for offset in range(rule["duration_years"]):
        out[START_YEAR + offset] = loss
        loss /= 2
    return out


def clean_html_text(value: str) -> str:
    value = re.sub(r"<br\s*/?>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", "", value)
    value = unescape(value)
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def extract_table_rows(path: Path, table_id: str) -> list[list[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    table_match = re.search(rf'<table[^>]*id="{table_id}"[^>]*>(.*?)</table>', text, flags=re.I | re.S)
    if not table_match:
        raise ValueError(f"Could not find table {table_id} in {path}")
    rows = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_match.group(1), flags=re.I | re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, flags=re.I | re.S)
        if cells:
            rows.append([clean_html_text(cell) for cell in cells])
    return rows


def parse_numeric(value: str) -> float:
    value = value.replace(",", "").replace("$", "").replace("%", "").strip()
    if not value or value in {"n/a", "N/A", "--"}:
        return float("nan")
    return float(value)


def build_tttr_state_table() -> pd.DataFrame:
    rows = extract_table_rows(EXEC_SUMMARY_PATH, "table4")
    records = []
    for cells in rows[1:]:
        if len(cells) < 5:
            continue
        state_name = cells[0]
        abbr = STATE_NAME_TO_ABBR.get(state_name)
        if not abbr or abbr in EXCLUDED_STATES:
            continue
        records.append(
            {
                "state_abbr": abbr,
                "tttr_2017": parse_numeric(cells[1]),
                "tttr_2018": parse_numeric(cells[2]),
                "tttr_2019": parse_numeric(cells[3]),
                "tttr_change_pct_2017_2019": parse_numeric(cells[4]),
            }
        )
    return pd.DataFrame(records)


def split_state_field(value: str) -> list[str]:
    parts = [part.strip().upper() for part in re.split(r"[/,]", value) if part.strip()]
    return [part for part in parts if part in STATE_ABBR_TO_FIPS and part not in EXCLUDED_STATES]


def build_bottleneck_state_table() -> pd.DataFrame:
    rows = extract_table_rows(APPENDIX_B_PATH, "table35")
    records = []
    for cells in rows[1:]:
        if len(cells) < 9:
            continue
        states = split_state_field(cells[4])
        if not states:
            continue
        dpm = parse_numeric(cells[7])
        congestion_cost = parse_numeric(cells[8])
        split_n = len(states)
        for state in states:
            records.append(
                {
                    "state_abbr": state,
                    "bottleneck_count": 1.0 / split_n,
                    "bottleneck_dpm": dpm / split_n,
                    "bottleneck_congestion_cost": congestion_cost / split_n,
                }
            )
    agg = pd.DataFrame(records).groupby("state_abbr", as_index=False).sum()
    return agg


def build_state_scores() -> pd.DataFrame:
    tttr = build_tttr_state_table()
    bottlenecks = build_bottleneck_state_table()
    scores = tttr.merge(bottlenecks, on="state_abbr", how="outer").fillna(0.0)
    scores["state_fips"] = scores["state_abbr"].map(STATE_ABBR_TO_FIPS)
    for source_col, score_col in [
        ("tttr_2019", "tttr_2019_rank_pct"),
        ("tttr_change_pct_2017_2019", "tttr_change_rank_pct"),
        ("bottleneck_count", "bottleneck_count_rank_pct"),
        ("bottleneck_dpm", "bottleneck_dpm_rank_pct"),
        ("bottleneck_congestion_cost", "bottleneck_cost_rank_pct"),
    ]:
        scores[score_col] = scores[source_col].rank(pct=True)
    score_cols = [
        "tttr_2019_rank_pct",
        "tttr_change_rank_pct",
        "bottleneck_count_rank_pct",
        "bottleneck_dpm_rank_pct",
        "bottleneck_cost_rank_pct",
    ]
    scores["domestic_bottleneck_score"] = scores[score_cols].mean(axis=1) * 100.0
    scores = scores.sort_values(
        ["domestic_bottleneck_score", "bottleneck_dpm", "tttr_2019"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return scores


def assign_cases(scores: pd.DataFrame) -> pd.DataFrame:
    selected = scores.head(5).copy()
    severities = ["severe", "medium", "medium", "mild", "mild"]
    reasons = [
        "top_domestic_bottleneck_score",
        "high_tttr_and_bottleneck_exposure",
        "high_tttr_and_bottleneck_exposure",
        "moderate_broad_congestion_exposure",
        "moderate_broad_congestion_exposure",
    ]
    selected["severity"] = severities[: len(selected)]
    selected["selection_reason"] = reasons[: len(selected)]
    selected["scenario_name"] = "domestic_bottleneck_state_blocking"
    return selected


def load_routes() -> pd.DataFrame:
    routes = pd.read_csv(
        BASE_ROUTE_PATH,
        dtype={"orig_state_fips": "string", "dest_state_fips": "string", "sctg2": "string"},
    )
    routes["orig_state_fips"] = routes["orig_state_fips"].astype(str).str.zfill(2)
    routes["dest_state_fips"] = routes["dest_state_fips"].astype(str).str.zfill(2)
    routes["prediction"] = pd.to_numeric(routes["prediction"], errors="coerce")
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["sctg2"] = routes["sctg2"].astype(str)
    return routes


def apply_state_blocking(shocked: pd.DataFrame, cases: pd.DataFrame) -> None:
    interstate_mask = shocked["orig_state_fips"] != shocked["dest_state_fips"]
    for row in cases.itertuples(index=False):
        state_mask = interstate_mask & (
            (shocked["orig_state_fips"] == row.state_fips) | (shocked["dest_state_fips"] == row.state_fips)
        )
        for year, loss in schedule(row.severity).items():
            mask = state_mask & (shocked["year"] == year)
            shocked.loc[mask, "prediction"] = shocked.loc[mask, "prediction"] * (1 - loss)
            shocked.loc[mask, "state_block_loss_applied"] = shocked.loc[mask, "state_block_loss_applied"] + loss


def build_hub_rankings(df: pd.DataFrame) -> pd.DataFrame:
    years = sorted(df["year"].unique().tolist())
    states = sorted(set(df["orig_state_fips"]) | set(df["dest_state_fips"]))
    rows = []
    for year in years:
        year_df = df.loc[df["year"] == year]
        for state in states:
            state_key = str(int(state)).zfill(2)
            out_inter = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "prediction"].sum()
            in_inter = year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "prediction"].sum()
            internal = year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] == state), "prediction"].sum()
            touch_total = out_inter + in_inter + internal
            partner_count = len(
                set(year_df.loc[(year_df["orig_state_fips"] == state) & (year_df["dest_state_fips"] != state), "dest_state_fips"].astype(str))
                | set(year_df.loc[(year_df["dest_state_fips"] == state) & (year_df["orig_state_fips"] != state), "orig_state_fips"].astype(str))
            )
            commodity_mask = (year_df["orig_state_fips"] == state) | (year_df["dest_state_fips"] == state)
            commodity_count = year_df.loc[commodity_mask & (year_df["prediction"] > 0), "sctg2"].nunique()
            rows.append(
                {
                    "year": year,
                    "state_fips": state,
                    "state_abbr": STATE_ABBR_MAP.get(state_key, state_key),
                    "touch_total": float(touch_total),
                    "partner_count": int(partner_count),
                    "commodity_count": int(commodity_count),
                    "internal_share": float(internal / touch_total) if touch_total > 0 else 0.0,
                    "interstate_share": float((out_inter + in_inter) / touch_total) if touch_total > 0 else 0.0,
                }
            )
    hub = pd.DataFrame(rows)
    base = hub.loc[hub["year"] == 2026, ["state_fips", "touch_total"]].rename(columns={"touch_total": "touch_2026"})
    last = hub.loc[hub["year"] == 2031, ["state_fips", "touch_total", "partner_count", "commodity_count", "internal_share", "interstate_share"]].rename(
        columns={
            "touch_total": "touch_2031",
            "partner_count": "partner_count_2031",
            "commodity_count": "commodity_count_2031",
            "internal_share": "internal_share_2031",
            "interstate_share": "interstate_share_2031",
        }
    )
    avg = hub.groupby(["state_fips", "state_abbr"], as_index=False)["touch_total"].mean().rename(columns={"touch_total": "avg_touch_2025_2031"})
    summary = avg.merge(base, on="state_fips", how="left").merge(last, on="state_fips", how="left")
    summary["touch_cagr_2026_2031"] = (summary["touch_2031"] / summary["touch_2026"]).pow(1 / 5) - 1
    for col_in, col_out in [
        ("touch_2031", "volume_rank_pct"),
        ("touch_cagr_2026_2031", "growth_rank_pct"),
        ("partner_count_2031", "partner_rank_pct"),
        ("commodity_count_2031", "commodity_rank_pct"),
    ]:
        summary[col_out] = summary[col_in].rank(pct=True)
    summary["hub_score"] = summary[["volume_rank_pct", "growth_rank_pct", "partner_rank_pct", "commodity_rank_pct"]].mean(axis=1) * 100.0
    summary = summary.sort_values(["hub_score", "touch_2031"], ascending=[False, False]).reset_index(drop=True)
    summary["hub_rank_2031"] = range(1, len(summary) + 1)
    summary["hub_type"] = summary["interstate_share_2031"].ge(0.40).map({True: "Backbone", False: "Internal"})
    return summary


def summarize_losses(base: pd.DataFrame, shocked: pd.DataFrame) -> dict:
    annual_base = base.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_total"})
    annual_shocked = shocked.groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shocked_total"})
    annual = annual_base.merge(annual_shocked, on="year", how="left")
    annual["loss_pct"] = (annual["base_total"] - annual["shocked_total"]) / annual["base_total"]

    interstate_base = base.loc[base["orig_state_fips"] != base["dest_state_fips"]].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "base_interstate"})
    interstate_shocked = shocked.loc[shocked["orig_state_fips"] != shocked["dest_state_fips"]].groupby("year", as_index=False)["prediction"].sum().rename(columns={"prediction": "shocked_interstate"})
    interstate = interstate_base.merge(interstate_shocked, on="year", how="left")
    interstate["loss_pct"] = (interstate["base_interstate"] - interstate["shocked_interstate"]) / interstate["base_interstate"]

    return {
        "scenario_name": "domestic_bottleneck_state_blocking",
        "cumulative_network_loss_pct_of_annual_sum": annual["loss_pct"].sum(),
        "peak_network_loss_pct": annual["loss_pct"].max(),
        "peak_network_loss_year": int(annual.loc[annual["loss_pct"].idxmax(), "year"]),
        "cumulative_interstate_loss_pct_of_annual_sum": interstate["loss_pct"].sum(),
        "peak_interstate_loss_pct": interstate["loss_pct"].max(),
        "peak_interstate_loss_year": int(interstate.loc[interstate["loss_pct"].idxmax(), "year"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = build_state_scores()
    cases = assign_cases(scores)
    routes = load_routes()
    must_have = set(pd.read_csv(MUST_HAVE_PATH)["state"].astype(str))

    shocked = routes.copy()
    shocked["state_block_loss_applied"] = 0.0
    apply_state_blocking(shocked, cases[["state_fips", "severity"]])

    rankings = build_hub_rankings(shocked)
    next15 = rankings.loc[~rankings["state_abbr"].isin(must_have)].head(TOP_N).copy()
    next15["next_rank"] = range(1, len(next15) + 1)
    losses = summarize_losses(routes, shocked)

    scores.to_csv(OUT_DIR / "domestic_bottleneck_state_scores.csv", index=False)
    cases.to_csv(OUT_DIR / "domestic_bottleneck_state_blocking_cases.csv", index=False)
    shocked.to_csv(OUT_DIR / "domestic_bottleneck_shocked_routes_2025_2031.csv", index=False)
    rankings.to_csv(OUT_DIR / "domestic_bottleneck_hub_rankings_2031.csv", index=False)
    next15.to_csv(OUT_DIR / "domestic_bottleneck_next15_excluding_must_have.csv", index=False)
    pd.DataFrame([losses]).to_csv(OUT_DIR / "domestic_bottleneck_summary.csv", index=False)

    lines = [
        "# Domestic Bottleneck Shock Reference",
        "",
        "This track is separated from natural-disaster and construction/infrastructure scenarios.",
        "State-level bottleneck exposure is scored from official FHWA freight mobility sources using:",
        "- 2019 TTTR index by State",
        "- 2017-2019 TTTR change by State",
        "- top freight bottleneck corridor count by State",
        "- top freight bottleneck delay-per-mile burden by State",
        "- top freight bottleneck congestion cost by State",
        "",
        "Selected blocking states:",
        "- " + ", ".join(f"{row.state_abbr} ({row.severity})" for row in cases.itertuples(index=False)),
        "",
        f"- cumulative network loss: {losses['cumulative_network_loss_pct_of_annual_sum']:.2%}",
        f"- peak network loss: {losses['peak_network_loss_pct']:.2%} in {losses['peak_network_loss_year']}",
        f"- cumulative interstate loss: {losses['cumulative_interstate_loss_pct_of_annual_sum']:.2%}",
        f"- peak interstate loss: {losses['peak_interstate_loss_pct']:.2%} in {losses['peak_interstate_loss_year']}",
        "",
        "Top 15 non-must-have candidates under this domestic bottleneck shock:",
        "- " + ", ".join(next15["state_abbr"].tolist()),
    ]
    (OUT_DIR / "DOMESTIC_BOTTLENECK_SHOCK_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "domestic_bottleneck_metadata.json").write_text(
        json.dumps(
            {
                "scores_csv": str(OUT_DIR / "domestic_bottleneck_state_scores.csv"),
                "cases_csv": str(OUT_DIR / "domestic_bottleneck_state_blocking_cases.csv"),
                "summary_csv": str(OUT_DIR / "domestic_bottleneck_summary.csv"),
                "next15_csv": str(OUT_DIR / "domestic_bottleneck_next15_excluding_must_have.csv"),
                "rankings_csv": str(OUT_DIR / "domestic_bottleneck_hub_rankings_2031.csv"),
                "reference_md": str(OUT_DIR / "DOMESTIC_BOTTLENECK_SHOCK_REFERENCE.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

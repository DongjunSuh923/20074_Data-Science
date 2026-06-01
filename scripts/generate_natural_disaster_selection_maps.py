from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import pandas as pd
import plotly.graph_objects as go


CASES_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "natural_disaster_state_blocking_cases.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "scenario_model" / "natural_disaster_capacity" / "simulation_results" / "natural_disaster_simulation_summary.csv"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "natural_disaster_visuals"

ALL_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","DC","FL","GA","HI","ID","IL","IN","IA","KS","KY","LA","ME","MD","MA",
    "MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX",
    "UT","VT","VA","WA","WV","WI","WY"
]

STATE_CENTROIDS = {
    "AL": (32.8, -86.8), "AZ": (34.2, -111.7), "AR": (34.9, -92.4), "CA": (37.2, -119.7),
    "CO": (39.0, -105.5), "CT": (41.6, -72.7), "DE": (39.0, -75.5), "DC": (38.9, -77.0),
    "FL": (28.4, -82.4), "GA": (32.7, -83.3), "ID": (44.2, -114.5), "IL": (40.0, -89.2),
    "IN": (39.9, -86.3), "IA": (42.1, -93.5), "KS": (38.5, -98.0), "KY": (37.5, -85.3),
    "LA": (31.2, -92.3), "ME": (45.3, -69.0), "MD": (39.0, -76.7), "MA": (42.2, -71.8),
    "MI": (44.3, -85.6), "MN": (46.0, -94.3), "MS": (32.7, -89.7), "MO": (38.5, -92.5),
    "MT": (46.9, -110.0), "NE": (41.5, -99.7), "NV": (39.3, -116.6), "NH": (43.7, -71.6),
    "NJ": (40.1, -74.5), "NM": (34.4, -106.1), "NY": (42.9, -75.5), "NC": (35.5, -79.0),
    "ND": (47.5, -100.5), "OH": (40.3, -82.8), "OK": (35.6, -97.5), "OR": (43.9, -120.6),
    "PA": (41.0, -77.5), "RI": (41.7, -71.5), "SC": (33.8, -80.9), "SD": (44.4, -100.2),
    "TN": (35.8, -86.4), "TX": (31.5, -99.3), "UT": (39.3, -111.7), "VT": (44.1, -72.7),
    "VA": (37.5, -78.7), "WA": (47.4, -120.7), "WV": (38.6, -80.6), "WI": (44.6, -89.6),
    "WY": (43.0, -107.6),
}

FIPS_TO_ABBR = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE", "11": "DC",
    "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT",
    "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", "56": "WY",
}

HAZARD_LABELS = {
    "snow_heat": "Snow / Heat",
    "tornado": "Tornado",
    "wildfire_smoke": "Wildfire / Smoke",
}

HAZARD_COLORS = {
    "snow_heat": {"none": "#E5E7EB", "mild": "#BFDBFE", "medium": "#60A5FA", "severe": "#1D4ED8"},
    "tornado": {"none": "#E5E7EB", "mild": "#FCD34D", "medium": "#F59E0B", "severe": "#B45309"},
    "wildfire_smoke": {"none": "#E5E7EB", "mild": "#FDBA74", "medium": "#F97316", "severe": "#C2410C"},
}

COMBINED_COLORS = {
    "Other": "#E5E7EB",
    "Snow / Heat only": "#60A5FA",
    "Tornado only": "#F59E0B",
    "Wildfire / Smoke only": "#F97316",
    "Multiple hazards": "#7C3AED",
}


def load_cases() -> pd.DataFrame:
    cases = pd.read_csv(CASES_PATH, dtype={"state_fips": "string"})
    cases["state_abbr"] = cases["state_fips"].astype(str).str.zfill(2).map(FIPS_TO_ABBR)
    return cases


def write_hazard_map(cases: pd.DataFrame, hazard: str, out_name: str, title: str) -> None:
    colors = HAZARD_COLORS[hazard]
    hazard_cases = cases.loc[cases["hazard"] == hazard, ["state_abbr", "severity"]].copy()
    sev_map = dict(zip(hazard_cases["state_abbr"], hazard_cases["severity"]))
    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df["severity"] = df["state_abbr"].map(lambda s: sev_map.get(s, "none"))

    fig = go.Figure()
    for idx, severity in enumerate(["none", "mild", "medium", "severe"]):
        sub = df.loc[df["severity"] == severity]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, colors[severity]], [1, colors[severity]]],
                showscale=False,
                showlegend=severity != "none",
                name=severity.capitalize(),
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + severity.capitalize() + "</extra>",
            )
        )
    labeled = sorted(set(hazard_cases["state_abbr"]) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labeled],
            lon=[STATE_CENTROIDS[s][1] for s in labeled],
            mode="text",
            text=labeled,
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=title,
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / out_name), include_plotlyjs="cdn")


def write_combined_map(cases: pd.DataFrame) -> None:
    grouped = cases.groupby("state_abbr")["hazard"].agg(list).reset_index()

    def classify(hazards: list[str]) -> str:
        unique = sorted(set(hazards))
        if len(unique) > 1:
            return "Multiple hazards"
        if unique[0] == "snow_heat":
            return "Snow / Heat only"
        if unique[0] == "tornado":
            return "Tornado only"
        return "Wildfire / Smoke only"

    grouped["category"] = grouped["hazard"].apply(classify)
    cat_map = dict(zip(grouped["state_abbr"], grouped["category"]))
    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df["category"] = df["state_abbr"].map(lambda s: cat_map.get(s, "Other"))

    fig = go.Figure()
    order = ["Other", "Snow / Heat only", "Tornado only", "Wildfire / Smoke only", "Multiple hazards"]
    for idx, category in enumerate(order):
        sub = df.loc[df["category"] == category]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[idx] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, COMBINED_COLORS[category]], [1, COMBINED_COLORS[category]]],
                showscale=False,
                showlegend=category != "Other",
                name=category,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + category + "</extra>",
            )
        )
    labeled = sorted(set(grouped["state_abbr"]) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labeled],
            lon=[STATE_CENTROIDS[s][1] for s in labeled],
            mode="text",
            text=labeled,
            textfont=dict(size=11, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Natural Disaster Track: Combined Selected States",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
    )
    fig.write_html(str(OUT_DIR / "01_natural_disaster_combined_selection_map.html"), include_plotlyjs="cdn")


def write_summary(cases: pd.DataFrame) -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    lines = [
        "# Natural Disaster Track Map Reference",
        "",
        "This folder contains actual U.S. state-boundary maps for the natural disaster track.",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.extend(
            [
                f"## {row.scenario_name}",
                "",
                f"- blocked states: {row.blocked_states}",
                f"- cumulative network loss: {row.cumulative_network_loss_pct_of_annual_sum:.2%}",
                f"- peak network loss: {row.peak_year_loss_pct:.2%} in {int(row.peak_year)}",
                "",
            ]
        )
    (OUT_DIR / "NATURAL_DISASTER_MAP_REFERENCE.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    write_combined_map(cases)
    write_hazard_map(cases, "snow_heat", "02_snow_heat_selection_map.html", "Natural Disaster Track: Snow / Heat Selection")
    write_hazard_map(cases, "tornado", "03_tornado_selection_map.html", "Natural Disaster Track: Tornado Selection")
    write_hazard_map(cases, "wildfire_smoke", "04_wildfire_smoke_selection_map.html", "Natural Disaster Track: Wildfire / Smoke Selection")
    write_summary(cases)
    metadata = {
        "combined_html": str(OUT_DIR / "01_natural_disaster_combined_selection_map.html"),
        "snow_heat_html": str(OUT_DIR / "02_snow_heat_selection_map.html"),
        "tornado_html": str(OUT_DIR / "03_tornado_selection_map.html"),
        "wildfire_smoke_html": str(OUT_DIR / "04_wildfire_smoke_selection_map.html"),
        "summary_md": str(OUT_DIR / "NATURAL_DISASTER_MAP_REFERENCE.md"),
    }
    (OUT_DIR / "natural_disaster_map_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

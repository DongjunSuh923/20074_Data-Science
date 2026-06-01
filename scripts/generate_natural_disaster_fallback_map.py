from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
WORK_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")

for dep_dir in [PROJECT_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import pandas as pd
import plotly.graph_objects as go


NEXT15_DIR = PROJECT_ROOT / "outputs" / "scenario_model" / "scenario_next15_hubs"
OUT_DIR = WORK_ROOT / "outputs" / "scenario_model" / "presentation_prep"

SCENARIO_FILES = {
    "snow_heat": NEXT15_DIR / "snow_heat_state_blocking_next15_excluding_must_have.csv",
    "tornado": NEXT15_DIR / "tornado_state_blocking_next15_excluding_must_have.csv",
    "wildfire_smoke": NEXT15_DIR / "wildfire_smoke_state_blocking_next15_excluding_must_have.csv",
}

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

STATE_KR = {
    "AL": "앨라배마", "AK": "알래스카", "AZ": "애리조나", "AR": "아칸소", "CA": "캘리포니아", "CO": "콜로라도",
    "CT": "코네티컷", "DE": "델라웨어", "DC": "워싱턴DC", "FL": "플로리다", "GA": "조지아", "HI": "하와이",
    "ID": "아이다호", "IL": "일리노이", "IN": "인디애나", "IA": "아이오와", "KS": "캔자스", "KY": "켄터키",
    "LA": "루이지애나", "ME": "메인", "MD": "메릴랜드", "MA": "매사추세츠", "MI": "미시간", "MN": "미네소타",
    "MS": "미시시피", "MO": "미주리", "MT": "몬태나", "NE": "네브래스카", "NV": "네바다", "NH": "뉴햄프셔",
    "NJ": "뉴저지", "NM": "뉴멕시코", "NY": "뉴욕", "NC": "노스캐롤라이나", "ND": "노스다코타", "OH": "오하이오",
    "OK": "오클라호마", "OR": "오리건", "PA": "펜실베이니아", "RI": "로드아일랜드", "SC": "사우스캐롤라이나",
    "SD": "사우스다코타", "TN": "테네시", "TX": "텍사스", "UT": "유타", "VT": "버몬트", "VA": "버지니아",
    "WA": "워싱턴", "WV": "웨스트버지니아", "WI": "위스콘신", "WY": "와이오밍",
}

COUNT_COLORS = {
    0: "#E5E7EB",
    1: "#BFDBFE",
    2: "#60A5FA",
    3: "#1D4ED8",
}


def load_repeat_counts() -> pd.DataFrame:
    rows: list[dict[str, str | int]] = []
    for scenario_name, path in SCENARIO_FILES.items():
        frame = pd.read_csv(path)
        top15 = frame.loc[frame["next_rank"] <= 15, ["state_abbr", "next_rank"]].copy()
        top15["scenario_name"] = scenario_name
        rows.extend(top15.to_dict("records"))

    combined = pd.DataFrame(rows)
    counts = (
        combined.groupby("state_abbr", as_index=False)
        .agg(
            repeat_count=("scenario_name", "nunique"),
            best_rank=("next_rank", "min"),
        )
    )
    df = pd.DataFrame({"state_abbr": ALL_STATES})
    df = df.merge(counts, on="state_abbr", how="left").fillna({"repeat_count": 0, "best_rank": 999})
    df["repeat_count"] = df["repeat_count"].astype(int)
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_repeat_counts()

    fig = go.Figure()
    for count in [0, 1, 2, 3]:
        sub = df.loc[df["repeat_count"] == count]
        name = {0: "Other", 1: "Appears in 1 scenario", 2: "Appears in 2 scenarios", 3: "Appears in 3 scenarios"}[count]
        fig.add_trace(
            go.Choropleth(
                locations=sub["state_abbr"],
                z=[count] * len(sub),
                locationmode="USA-states",
                colorscale=[[0, COUNT_COLORS[count]], [1, COUNT_COLORS[count]]],
                showscale=False,
                showlegend=count != 0,
                name=name,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + name + "</extra>",
            )
        )

    labeled = sorted(set(df.loc[df["repeat_count"] >= 2, "state_abbr"]) & set(STATE_CENTROIDS))
    fig.add_trace(
        go.Scattergeo(
            lat=[STATE_CENTROIDS[s][0] for s in labeled],
            lon=[STATE_CENTROIDS[s][1] for s in labeled],
            mode="text",
            text=[f"{s} ({STATE_KR.get(s, s)})" for s in labeled],
            textfont=dict(size=12, color="#111827", family="Arial Black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # Explicitly call out AR as a fallback state.
    if "AR" in STATE_CENTROIDS:
        fig.add_trace(
            go.Scattergeo(
                lat=[STATE_CENTROIDS["AR"][0]],
                lon=[STATE_CENTROIDS["AR"][1]],
                mode="markers+text",
                text=[f"AR ({STATE_KR['AR']})"],
                textposition="bottom center",
                marker=dict(size=10, color="#F28E2B", line=dict(width=1.5, color="white")),
                textfont=dict(size=13, color="#B45309", family="Arial Black"),
                hovertemplate="AR: fallback candidate<extra></extra>",
                name="AR highlight",
                showlegend=False,
            )
        )

    fig.update_layout(
        title="Natural Disaster Track: Repeated Fallback Candidates",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5),
        annotations=[
            dict(
                x=0.5,
                y=0.02,
                xref="paper",
                yref="paper",
                text="AR appears in snow/heat and tornado as a south-central fallback candidate.",
                showarrow=False,
                font=dict(size=11, color="#374151"),
            )
        ],
    )

    html_path = OUT_DIR / "06_natural_disaster_fallback_map.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")


if __name__ == "__main__":
    main()

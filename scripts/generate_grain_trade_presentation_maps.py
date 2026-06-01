from __future__ import annotations

from pathlib import Path
import sys

DOWNLOAD_ROOT = Path(r"C:\Users\서동준\Downloads\FAF5.7.1_2018-2024")
IDEA_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")

for dep_dir in [DOWNLOAD_ROOT / ".pydeps", IDEA_ROOT / ".pydeps"]:
    if dep_dir.exists() and str(dep_dir) not in sys.path:
        sys.path.insert(0, str(dep_dir))

import plotly.graph_objects as go


OUT_DIR = DOWNLOAD_ROOT / "outputs" / "scenario_model" / "presentation_prep"

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


def add_choropleth_categories(fig: go.Figure, category_map: dict[str, str], color_map: dict[str, str], order: list[str]) -> None:
    for idx, category in enumerate(order):
        locations = [s for s in ALL_STATES if category_map.get(s, "Other") == category]
        fig.add_trace(
            go.Choropleth(
                locations=locations,
                z=[idx] * len(locations),
                locationmode="USA-states",
                colorscale=[[0, color_map[category]], [1, color_map[category]]],
                showscale=False,
                showlegend=category != "Other",
                name=category,
                marker_line_color="white",
                marker_line_width=0.8,
                hovertemplate="%{location}<extra>" + category + "</extra>",
            )
        )


def make_grain_map() -> None:
    shock_categories = {
        "IA": "Severe",
        "TX": "Medium", "NE": "Medium",
        "IL": "Mild", "MN": "Mild",
        "KS": "Support core",
        "WA": "Export axis", "OR": "Export axis",
    }
    color_map = {
        "Other": "#E5E7EB",
        "Mild": "#BFDBFE",
        "Medium": "#60A5FA",
        "Severe": "#1D4ED8",
        "Support core": "#F59E0B",
        "Export axis": "#16A34A",
    }
    order = ["Other", "Mild", "Medium", "Severe", "Support core", "Export axis"]

    fig = go.Figure()
    add_choropleth_categories(fig, shock_categories, color_map, order)

    labeled = sorted(set(shock_categories) & set(STATE_CENTROIDS))
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
        title="Grain Supply Shock: Shock States and Grain Support Axis",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.write_html(str(OUT_DIR / "10_grain_supply_presentation_map.html"), include_plotlyjs="cdn")


def make_trade_map() -> None:
    categories = {
        "CA": "Severe",
        "GA": "Medium", "NJ": "Medium",
        "MI": "Mild", "IL": "Mild",
        "NC": "EC connector", "VA": "EC connector",
    }
    color_map = {
        "Other": "#E5E7EB",
        "Mild": "#FCD34D",
        "Medium": "#F59E0B",
        "Severe": "#B45309",
        "EC connector": "#16A34A",
    }
    order = ["Other", "Mild", "Medium", "Severe", "EC connector"]

    fig = go.Figure()
    add_choropleth_categories(fig, categories, color_map, order)

    labeled = sorted(set(categories) & set(STATE_CENTROIDS))
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
        title="Enhanced Trade Shock: Gateway States and East Coast Connectors",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="#F8FAFC"),
        margin=dict(l=20, r=20, t=70, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.01, xanchor="center", x=0.5, font=dict(size=10)),
    )
    fig.write_html(str(OUT_DIR / "11_trade_external_presentation_map.html"), include_plotlyjs="cdn")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_grain_map()
    make_trade_map()


if __name__ == "__main__":
    main()

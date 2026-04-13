from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


INPUT_PATH = PROJECT_ROOT / "outputs" / "preprocessing_truck_only" / "route_yearly_summary_for_map.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "maps"
YEAR = 2024

STATE_INFO = {
    "01": ("AL", "Alabama", 32.806671, -86.791130),
    "02": ("AK", "Alaska", 61.370716, -152.404419),
    "04": ("AZ", "Arizona", 33.729759, -111.431221),
    "05": ("AR", "Arkansas", 34.969704, -92.373123),
    "06": ("CA", "California", 36.116203, -119.681564),
    "08": ("CO", "Colorado", 39.059811, -105.311104),
    "09": ("CT", "Connecticut", 41.597782, -72.755371),
    "10": ("DE", "Delaware", 39.318523, -75.507141),
    "11": ("DC", "District of Columbia", 38.9072, -77.0369),
    "12": ("FL", "Florida", 27.766279, -81.686783),
    "13": ("GA", "Georgia", 33.040619, -83.643074),
    "15": ("HI", "Hawaii", 21.094318, -157.498337),
    "16": ("ID", "Idaho", 44.240459, -114.478828),
    "17": ("IL", "Illinois", 40.349457, -88.986137),
    "18": ("IN", "Indiana", 39.849426, -86.258278),
    "19": ("IA", "Iowa", 42.011539, -93.210526),
    "20": ("KS", "Kansas", 38.5266, -96.726486),
    "21": ("KY", "Kentucky", 37.66814, -84.670067),
    "22": ("LA", "Louisiana", 31.169546, -91.867805),
    "23": ("ME", "Maine", 44.693947, -69.381927),
    "24": ("MD", "Maryland", 39.063946, -76.802101),
    "25": ("MA", "Massachusetts", 42.230171, -71.530106),
    "26": ("MI", "Michigan", 43.326618, -84.536095),
    "27": ("MN", "Minnesota", 45.694454, -93.900192),
    "28": ("MS", "Mississippi", 32.741646, -89.678696),
    "29": ("MO", "Missouri", 38.456085, -92.288368),
    "30": ("MT", "Montana", 46.921925, -110.454353),
    "31": ("NE", "Nebraska", 41.12537, -98.268082),
    "32": ("NV", "Nevada", 38.313515, -117.055374),
    "33": ("NH", "New Hampshire", 43.452492, -71.563896),
    "34": ("NJ", "New Jersey", 40.298904, -74.521011),
    "35": ("NM", "New Mexico", 34.840515, -106.248482),
    "36": ("NY", "New York", 42.165726, -74.948051),
    "37": ("NC", "North Carolina", 35.630066, -79.806419),
    "38": ("ND", "North Dakota", 47.528912, -99.784012),
    "39": ("OH", "Ohio", 40.388783, -82.764915),
    "40": ("OK", "Oklahoma", 35.565342, -96.928917),
    "41": ("OR", "Oregon", 44.572021, -122.070938),
    "42": ("PA", "Pennsylvania", 40.590752, -77.209755),
    "44": ("RI", "Rhode Island", 41.680893, -71.51178),
    "45": ("SC", "South Carolina", 33.856892, -80.945007),
    "46": ("SD", "South Dakota", 44.299782, -99.438828),
    "47": ("TN", "Tennessee", 35.747845, -86.692345),
    "48": ("TX", "Texas", 31.054487, -97.563461),
    "49": ("UT", "Utah", 40.150032, -111.862434),
    "50": ("VT", "Vermont", 44.045876, -72.710686),
    "51": ("VA", "Virginia", 37.769337, -78.169968),
    "53": ("WA", "Washington", 47.400902, -121.490494),
    "54": ("WV", "West Virginia", 38.491226, -80.954453),
    "55": ("WI", "Wisconsin", 44.268543, -89.616508),
    "56": ("WY", "Wyoming", 42.755966, -107.30249),
}


def prepare_state_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    year_df = df.loc[df["year"] == year].copy()

    origin = (
        year_df.groupby(["orig_state_fips", "orig_state_name"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"orig_state_fips": "state_fips", "orig_state_name": "state_name", "total_tons": "origin_tons"})
    )
    dest = (
        year_df.groupby(["dest_state_fips", "dest_state_name"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"dest_state_fips": "state_fips", "dest_state_name": "state_name", "total_tons": "destination_tons"})
    )

    state_df = origin.merge(dest, on=["state_fips", "state_name"], how="outer").fillna(0)
    state_df["net_destination_minus_origin"] = state_df["destination_tons"] - state_df["origin_tons"]
    state_df["state_abbr"] = state_df["state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[0])
    return state_df.sort_values("state_abbr")


def save_state_tables(state_df: pd.DataFrame, year: int) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    state_df.to_csv(OUTPUT_DIR / f"state_flows_{year}.csv", index=False)


def make_choropleth(state_df: pd.DataFrame, value_col: str, title: str, filename: str, scale: str) -> None:
    fig = px.choropleth(
        state_df,
        locations="state_abbr",
        locationmode="USA-states",
        color=value_col,
        scope="usa",
        hover_name="state_name",
        hover_data={
            "state_abbr": True,
            "origin_tons": ":,.0f",
            "destination_tons": ":,.0f",
            "net_destination_minus_origin": ":,.0f",
            value_col: ":,.0f",
        },
        color_continuous_scale=scale,
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        coloraxis_colorbar_title="Tons",
    )
    fig.write_html(str(OUTPUT_DIR / filename), include_plotlyjs="cdn")


def make_corridor_map(df: pd.DataFrame, year: int, top_n: int = 50) -> pd.DataFrame:
    year_df = df.loc[df["year"] == year].copy()
    state_routes = (
        year_df.groupby(
            [
                "orig_state_fips",
                "orig_state_name",
                "dest_state_fips",
                "dest_state_name",
            ],
            as_index=False,
        )["total_tons"]
        .sum()
        .sort_values("total_tons", ascending=False)
    )
    state_routes = state_routes.loc[state_routes["orig_state_fips"] != state_routes["dest_state_fips"]].head(top_n).copy()

    state_routes["orig_abbr"] = state_routes["orig_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[0])
    state_routes["dest_abbr"] = state_routes["dest_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[0])
    state_routes["orig_lat"] = state_routes["orig_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[2])
    state_routes["orig_lon"] = state_routes["orig_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[3])
    state_routes["dest_lat"] = state_routes["dest_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[2])
    state_routes["dest_lon"] = state_routes["dest_state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ("", "", None, None))[3])
    state_routes = state_routes.dropna().copy()
    state_routes["route_label"] = state_routes["orig_abbr"] + " -> " + state_routes["dest_abbr"]

    max_tons = state_routes["total_tons"].max() if not state_routes.empty else 1
    fig = go.Figure()

    for _, row in state_routes.iterrows():
        width = 1 + 7 * (row["total_tons"] / max_tons)
        fig.add_trace(
            go.Scattergeo(
                lon=[row["orig_lon"], row["dest_lon"]],
                lat=[row["orig_lat"], row["dest_lat"]],
                mode="lines",
                line=dict(width=width, color="rgba(214, 39, 40, 0.45)"),
                hoverinfo="text",
                text=f"{row['route_label']}<br>Tons: {row['total_tons']:,.0f}",
                showlegend=False,
            )
        )

    node_states = pd.concat(
        [
            state_routes[["orig_abbr", "orig_state_name", "orig_lat", "orig_lon"]].rename(
                columns={
                    "orig_abbr": "state_abbr",
                    "orig_state_name": "state_name",
                    "orig_lat": "lat",
                    "orig_lon": "lon",
                }
            ),
            state_routes[["dest_abbr", "dest_state_name", "dest_lat", "dest_lon"]].rename(
                columns={
                    "dest_abbr": "state_abbr",
                    "dest_state_name": "state_name",
                    "dest_lat": "lat",
                    "dest_lon": "lon",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()

    fig.add_trace(
        go.Scattergeo(
            lon=node_states["lon"],
            lat=node_states["lat"],
            mode="markers+text",
            text=node_states["state_abbr"],
            textposition="top center",
            marker=dict(size=7, color="#1f77b4", line=dict(width=0.5, color="white")),
            hoverinfo="text",
            hovertext=node_states["state_name"],
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Top {top_n} Interstate Truck Corridors ({year})",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="rgb(243,243,243)"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.write_html(str(OUTPUT_DIR / f"top_interstate_corridors_{year}.html"), include_plotlyjs="cdn")
    state_routes.to_csv(OUTPUT_DIR / f"top_interstate_corridors_{year}.csv", index=False)
    return state_routes


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT_PATH, dtype={"orig_state_fips": str, "dest_state_fips": str, "dms_orig": str, "dms_dest": str})
    state_df = prepare_state_year(df, YEAR)
    save_state_tables(state_df, YEAR)

    make_choropleth(
        state_df,
        "origin_tons",
        f"Truck Freight Origin Tons by State ({YEAR})",
        f"state_origin_tons_{YEAR}.html",
        "Blues",
    )
    make_choropleth(
        state_df,
        "destination_tons",
        f"Truck Freight Destination Tons by State ({YEAR})",
        f"state_destination_tons_{YEAR}.html",
        "Greens",
    )
    make_choropleth(
        state_df,
        "net_destination_minus_origin",
        f"Truck Freight Net Inflow by State ({YEAR})",
        f"state_net_inflow_{YEAR}.html",
        "RdBu",
    )
    make_corridor_map(df, YEAR, top_n=50)


if __name__ == "__main__":
    main()

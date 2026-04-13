from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
ROUTE_PATH = PREPROCESS_DIR / "route_yearly_summary_for_map.csv"
CENTROID_PATH = MAP_DIR / "dms_zone_centroids.csv"
YEARS = list(range(2018, 2025))
TOP_N = 60

STATE_INFO = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT", "10": "DE",
    "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", "34": "NJ", "35": "NM",
    "36": "NY", "37": "NC", "38": "ND", "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI",
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = pd.read_csv(
        ROUTE_PATH,
        dtype={"dms_orig": str, "dms_dest": str, "orig_state_fips": str, "dest_state_fips": str},
    )
    centroids = pd.read_csv(CENTROID_PATH, dtype={"dms_code": str})
    return routes, centroids


def make_state_animation(routes: pd.DataFrame) -> None:
    origin = (
        routes.groupby(["year", "orig_state_fips", "orig_state_name"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"orig_state_fips": "state_fips", "orig_state_name": "state_name", "total_tons": "origin_tons"})
    )
    dest = (
        routes.groupby(["year", "dest_state_fips", "dest_state_name"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"dest_state_fips": "state_fips", "dest_state_name": "state_name", "total_tons": "destination_tons"})
    )
    state_year = origin.merge(dest, on=["year", "state_fips", "state_name"], how="outer").fillna(0)
    state_year["net_destination_minus_origin"] = state_year["destination_tons"] - state_year["origin_tons"]
    state_year["state_abbr"] = state_year["state_fips"].map(lambda x: STATE_INFO.get(str(x).zfill(2), ""))
    state_year.to_csv(MAP_DIR / "state_flows_2018_2024.csv", index=False)

    for value_col, title, filename, scale in [
        ("origin_tons", "Truck Freight Origin Tons by State", "state_origin_tons_2018_2024.html", "Blues"),
        ("destination_tons", "Truck Freight Destination Tons by State", "state_destination_tons_2018_2024.html", "Greens"),
        ("net_destination_minus_origin", "Truck Freight Net Inflow by State", "state_net_inflow_2018_2024.html", "RdBu"),
    ]:
        fig = px.choropleth(
            state_year,
            locations="state_abbr",
            locationmode="USA-states",
            color=value_col,
            animation_frame="year",
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
            title=title + " (2018-2024)",
        )
        fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=60, b=20))
        fig.write_html(str(MAP_DIR / filename), include_plotlyjs="cdn")


def make_dms_animation(routes: pd.DataFrame, centroids: pd.DataFrame) -> None:
    route_frames = []
    node_frames = []

    for year in YEARS:
        top_routes = (
            routes.loc[(routes["year"] == year) & (routes["dms_orig"] != routes["dms_dest"])]
            .sort_values("total_tons", ascending=False)
            .head(TOP_N)
            .copy()
        )
        top_routes = top_routes.merge(
            centroids[["dms_code", "latitude", "longitude"]].rename(
                columns={"dms_code": "dms_orig", "latitude": "orig_lat", "longitude": "orig_lon"}
            ),
            on="dms_orig",
            how="left",
        ).merge(
            centroids[["dms_code", "latitude", "longitude"]].rename(
                columns={"dms_code": "dms_dest", "latitude": "dest_lat", "longitude": "dest_lon"}
            ),
            on="dms_dest",
            how="left",
        )
        top_routes = top_routes.dropna(subset=["orig_lat", "orig_lon", "dest_lat", "dest_lon"]).copy()
        top_routes["route_label"] = top_routes["orig_zone_name"] + " -> " + top_routes["dest_zone_name"]
        top_routes["year"] = year
        route_frames.append(top_routes)

        nodes = pd.concat(
            [
                top_routes[["dms_orig", "orig_zone_name", "orig_state_name", "orig_lat", "orig_lon", "total_tons"]].rename(
                    columns={
                        "dms_orig": "dms_code",
                        "orig_zone_name": "zone_name",
                        "orig_state_name": "state_name",
                        "orig_lat": "lat",
                        "orig_lon": "lon",
                        "total_tons": "route_tons",
                    }
                ),
                top_routes[["dms_dest", "dest_zone_name", "dest_state_name", "dest_lat", "dest_lon", "total_tons"]].rename(
                    columns={
                        "dms_dest": "dms_code",
                        "dest_zone_name": "zone_name",
                        "dest_state_name": "state_name",
                        "dest_lat": "lat",
                        "dest_lon": "lon",
                        "total_tons": "route_tons",
                    }
                ),
            ],
            ignore_index=True,
        )
        nodes = nodes.groupby(["dms_code", "zone_name", "state_name", "lat", "lon"], as_index=False)["route_tons"].sum()
        nodes["year"] = year
        node_frames.append(nodes)

    route_df = pd.concat(route_frames, ignore_index=True)
    node_df = pd.concat(node_frames, ignore_index=True)
    route_df.to_csv(MAP_DIR / "top_dms_corridors_2018_2024.csv", index=False)
    node_df.to_csv(MAP_DIR / "top_dms_corridor_nodes_2018_2024.csv", index=False)

    max_tons = route_df["total_tons"].max() if not route_df.empty else 1
    max_node_tons = node_df["route_tons"].max() if not node_df.empty else 1

    frames = []
    for year in YEARS:
        yr_routes = route_df.loc[route_df["year"] == year]
        yr_nodes = node_df.loc[node_df["year"] == year]
        data = []
        for _, row in yr_routes.iterrows():
            width = 1 + 7 * (row["total_tons"] / max_tons)
            data.append(
                go.Scattergeo(
                    lon=[row["orig_lon"], row["dest_lon"]],
                    lat=[row["orig_lat"], row["dest_lat"]],
                    mode="lines",
                    line=dict(width=width, color="rgba(214, 39, 40, 0.35)"),
                    text=f"{row['route_label']}<br>Tons: {row['total_tons']:,.0f}",
                    hoverinfo="text",
                    showlegend=False,
                )
            )
        data.append(
            go.Scattergeo(
                lon=yr_nodes["lon"],
                lat=yr_nodes["lat"],
                mode="markers",
                marker=dict(
                    size=(yr_nodes["route_tons"] / max_node_tons * 20).clip(lower=5),
                    color=yr_nodes["route_tons"],
                    colorscale="Blues",
                    line=dict(width=0.5, color="white"),
                    opacity=0.85,
                    showscale=(year == YEARS[0]),
                    colorbar=dict(title="Connected tons"),
                ),
                text=yr_nodes["zone_name"] + ", " + yr_nodes["state_name"],
                hovertemplate="%{text}<br>Connected tons: %{marker.color:,.0f}<extra></extra>",
                showlegend=False,
            )
        )
        frames.append(go.Frame(data=data, name=str(year)))

    initial_year = YEARS[0]
    fig = go.Figure(frames=frames)
    fig.add_traces(frames[0].data)
    fig.update_layout(
        title=f"Top {TOP_N} DMS Truck Corridors (2018-2024)",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="rgb(245,245,245)"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.05,
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 800, "redraw": True}, "fromcurrent": True}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Year: "},
                "pad": {"t": 35},
                "steps": [
                    {
                        "label": str(year),
                        "method": "animate",
                        "args": [[str(year)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    }
                    for year in YEARS
                ],
            }
        ],
    )
    fig.write_html(str(MAP_DIR / "top_dms_corridors_2018_2024.html"), include_plotlyjs="cdn")


def main() -> None:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    routes, centroids = load_data()
    make_state_animation(routes)
    make_dms_animation(routes, centroids)


if __name__ == "__main__":
    main()

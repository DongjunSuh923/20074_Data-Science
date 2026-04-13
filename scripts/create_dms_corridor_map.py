from __future__ import annotations

import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import pandas as pd
import plotly.graph_objects as go
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim


PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only"
MAP_DIR = PROJECT_ROOT / "outputs" / "maps"
LOOKUP_PATH = PREPROCESS_DIR / "lookups" / "domestic_zone_lookup.csv"
ROUTE_PATH = PREPROCESS_DIR / "route_yearly_summary_for_map.csv"
CENTROID_CACHE_PATH = MAP_DIR / "dms_zone_centroids.csv"
YEAR = 2024
TOP_N = 100

STATE_INFO = {
    "AL": (32.806671, -86.791130),
    "AK": (61.370716, -152.404419),
    "AZ": (33.729759, -111.431221),
    "AR": (34.969704, -92.373123),
    "CA": (36.116203, -119.681564),
    "CO": (39.059811, -105.311104),
    "CT": (41.597782, -72.755371),
    "DE": (39.318523, -75.507141),
    "DC": (38.9072, -77.0369),
    "FL": (27.766279, -81.686783),
    "GA": (33.040619, -83.643074),
    "HI": (21.094318, -157.498337),
    "ID": (44.240459, -114.478828),
    "IL": (40.349457, -88.986137),
    "IN": (39.849426, -86.258278),
    "IA": (42.011539, -93.210526),
    "KS": (38.5266, -96.726486),
    "KY": (37.66814, -84.670067),
    "LA": (31.169546, -91.867805),
    "ME": (44.693947, -69.381927),
    "MD": (39.063946, -76.802101),
    "MA": (42.230171, -71.530106),
    "MI": (43.326618, -84.536095),
    "MN": (45.694454, -93.900192),
    "MS": (32.741646, -89.678696),
    "MO": (38.456085, -92.288368),
    "MT": (46.921925, -110.454353),
    "NE": (41.12537, -98.268082),
    "NV": (38.313515, -117.055374),
    "NH": (43.452492, -71.563896),
    "NJ": (40.298904, -74.521011),
    "NM": (34.840515, -106.248482),
    "NY": (42.165726, -74.948051),
    "NC": (35.630066, -79.806419),
    "ND": (47.528912, -99.784012),
    "OH": (40.388783, -82.764915),
    "OK": (35.565342, -96.928917),
    "OR": (44.572021, -122.070938),
    "PA": (40.590752, -77.209755),
    "RI": (41.680893, -71.51178),
    "SC": (33.856892, -80.945007),
    "SD": (44.299782, -99.438828),
    "TN": (35.747845, -86.692345),
    "TX": (31.054487, -97.563461),
    "UT": (40.150032, -111.862434),
    "VT": (44.045876, -72.710686),
    "VA": (37.769337, -78.169968),
    "WA": (47.400902, -121.490494),
    "WV": (38.491226, -80.954453),
    "WI": (44.268543, -89.616508),
    "WY": (42.755966, -107.302490),
}


def zone_query(short_desc: str, long_desc: str, state_name: str) -> str:
    short_desc = str(short_desc).strip()
    long_desc = str(long_desc).strip()
    state_name = str(state_name).strip()

    if short_desc.startswith("Rest of "):
        return f"{state_name}, USA"
    if short_desc == state_name:
        return f"{state_name}, USA"

    part_match = re.search(r"\(([A-Z]{2}) Part\)", short_desc)
    preferred_state = part_match.group(1) if part_match else None

    cleaned = re.sub(r"\s*\([^)]+\)", "", short_desc)
    m = re.match(r"(.+?)\s([A-Z]{2}(?:-[A-Z]{2})*)$", cleaned)
    if m:
        city_part = m.group(1)
        state_part = preferred_state or m.group(2).split("-")[0]
        return f"{city_part}, {state_part}, USA"

    city_guess = long_desc.split(",")[0]
    if preferred_state:
        return f"{city_guess}, {preferred_state}, USA"
    return f"{city_guess}, {state_name}, USA"


def state_abbr_from_short_desc(short_desc: str, state_name: str) -> str | None:
    short_desc = str(short_desc).strip()
    part_match = re.search(r"\(([A-Z]{2}) Part\)", short_desc)
    if part_match:
        return part_match.group(1)
    m = re.match(r"(.+?)\s([A-Z]{2})(?:-[A-Z]{2})*$", re.sub(r"\s*\([^)]+\)", "", short_desc))
    if m:
        return m.group(2)
    for abbr in STATE_INFO:
        if state_name.lower().startswith(
            {
                "AL": "alabama",
                "AK": "alaska",
                "AZ": "arizona",
                "AR": "arkansas",
                "CA": "california",
                "CO": "colorado",
                "CT": "connecticut",
                "DE": "delaware",
                "DC": "district of columbia",
                "FL": "florida",
                "GA": "georgia",
                "HI": "hawaii",
                "ID": "idaho",
                "IL": "illinois",
                "IN": "indiana",
                "IA": "iowa",
                "KS": "kansas",
                "KY": "kentucky",
                "LA": "louisiana",
                "ME": "maine",
                "MD": "maryland",
                "MA": "massachusetts",
                "MI": "michigan",
                "MN": "minnesota",
                "MS": "mississippi",
                "MO": "missouri",
                "MT": "montana",
                "NE": "nebraska",
                "NV": "nevada",
                "NH": "new hampshire",
                "NJ": "new jersey",
                "NM": "new mexico",
                "NY": "new york",
                "NC": "north carolina",
                "ND": "north dakota",
                "OH": "ohio",
                "OK": "oklahoma",
                "OR": "oregon",
                "PA": "pennsylvania",
                "RI": "rhode island",
                "SC": "south carolina",
                "SD": "south dakota",
                "TN": "tennessee",
                "TX": "texas",
                "UT": "utah",
                "VT": "vermont",
                "VA": "virginia",
                "WA": "washington",
                "WV": "west virginia",
                "WI": "wisconsin",
                "WY": "wyoming",
            }[abbr]
        ):
            return abbr
    return None


def build_centroid_cache() -> pd.DataFrame:
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    zones = pd.read_csv(LOOKUP_PATH, dtype={"dms_code": str, "state_fips": str})

    existing = pd.read_csv(CENTROID_CACHE_PATH, dtype={"dms_code": str}) if CENTROID_CACHE_PATH.exists() else pd.DataFrame()
    existing_codes = set(existing["dms_code"]) if not existing.empty else set()

    geolocator = Nominatim(user_agent="faf5_dms_corridor_mapper")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1, swallow_exceptions=True)

    new_rows = []
    for _, row in zones.iterrows():
        dms_code = row["dms_code"]
        if dms_code in existing_codes:
            continue

        query = zone_query(row["dms_short_desc"], row["dms_long_desc"], row["state_name"])
        lat = None
        lon = None
        source = "geocoded"
        location = geocode(query, timeout=20)
        if location is not None:
            lat = location.latitude
            lon = location.longitude
        else:
            state_abbr = state_abbr_from_short_desc(row["dms_short_desc"], row["state_name"])
            if state_abbr in STATE_INFO:
                lat, lon = STATE_INFO[state_abbr]
                source = "state_centroid_fallback"

        new_rows.append(
            {
                "dms_code": dms_code,
                "dms_short_desc": row["dms_short_desc"],
                "dms_long_desc": row["dms_long_desc"],
                "state_name": row["state_name"],
                "query": query,
                "latitude": lat,
                "longitude": lon,
                "source": source,
            }
        )
        print(f"geocoded {dms_code} -> {query} ({source})")

    if new_rows:
        all_rows = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
        all_rows.to_csv(CENTROID_CACHE_PATH, index=False)
    else:
        all_rows = existing

    return all_rows


def build_dms_corridor_outputs(year: int, top_n: int) -> None:
    routes = pd.read_csv(
        ROUTE_PATH,
        dtype={"dms_orig": str, "dms_dest": str, "orig_state_fips": str, "dest_state_fips": str},
    )
    centroids = build_centroid_cache()

    year_routes = routes.loc[(routes["year"] == year) & (routes["dms_orig"] != routes["dms_dest"])].copy()
    zone_totals = (
        year_routes.groupby(["dms_orig", "orig_zone_name", "orig_state_name"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"dms_orig": "dms_code", "orig_zone_name": "zone_name", "orig_state_name": "state_name"})
    )
    zone_in = (
        year_routes.groupby(["dms_dest"], as_index=False)["total_tons"]
        .sum()
        .rename(columns={"dms_dest": "dms_code", "total_tons": "inbound_tons"})
    )
    zone_totals = zone_totals.rename(columns={"total_tons": "outbound_tons"}).merge(zone_in, on="dms_code", how="outer").fillna(0)
    zone_totals["hub_tons"] = zone_totals["outbound_tons"] + zone_totals["inbound_tons"]
    zone_totals = zone_totals.merge(centroids[["dms_code", "latitude", "longitude", "source"]], on="dms_code", how="left")
    zone_totals.to_csv(MAP_DIR / f"dms_zone_hubs_{year}.csv", index=False)

    top_routes = (
        year_routes.sort_values("total_tons", ascending=False)
        .head(top_n)
        .copy()
    )
    top_routes = top_routes.merge(
        centroids[["dms_code", "latitude", "longitude", "source"]].rename(
            columns={
                "dms_code": "dms_orig",
                "latitude": "orig_lat",
                "longitude": "orig_lon",
                "source": "orig_coord_source",
            }
        ),
        on="dms_orig",
        how="left",
    ).merge(
        centroids[["dms_code", "latitude", "longitude", "source"]].rename(
            columns={
                "dms_code": "dms_dest",
                "latitude": "dest_lat",
                "longitude": "dest_lon",
                "source": "dest_coord_source",
            }
        ),
        on="dms_dest",
        how="left",
    )
    top_routes = top_routes.dropna(subset=["orig_lat", "orig_lon", "dest_lat", "dest_lon"]).copy()
    top_routes["route_label"] = top_routes["orig_zone_name"] + " -> " + top_routes["dest_zone_name"]
    top_routes.to_csv(MAP_DIR / f"top_dms_corridors_{year}.csv", index=False)

    fig = go.Figure()
    max_tons = top_routes["total_tons"].max() if not top_routes.empty else 1
    for _, row in top_routes.iterrows():
        width = 1 + 8 * (row["total_tons"] / max_tons)
        fig.add_trace(
            go.Scattergeo(
                lon=[row["orig_lon"], row["dest_lon"]],
                lat=[row["orig_lat"], row["dest_lat"]],
                mode="lines",
                line=dict(width=width, color="rgba(214, 39, 40, 0.38)"),
                text=(
                    f"{row['route_label']}<br>"
                    f"Tons: {row['total_tons']:,.0f}<br>"
                    f"Current Value: {row['total_current_value']:,.0f}"
                ),
                hoverinfo="text",
                showlegend=False,
            )
        )

    node_df = pd.concat(
        [
            top_routes[["dms_orig", "orig_zone_name", "orig_state_name", "orig_lat", "orig_lon"]].rename(
                columns={
                    "dms_orig": "dms_code",
                    "orig_zone_name": "zone_name",
                    "orig_state_name": "state_name",
                    "orig_lat": "lat",
                    "orig_lon": "lon",
                }
            ),
            top_routes[["dms_dest", "dest_zone_name", "dest_state_name", "dest_lat", "dest_lon"]].rename(
                columns={
                    "dms_dest": "dms_code",
                    "dest_zone_name": "zone_name",
                    "dest_state_name": "state_name",
                    "dest_lat": "lat",
                    "dest_lon": "lon",
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    node_df = node_df.merge(zone_totals[["dms_code", "hub_tons"]], on="dms_code", how="left")

    fig.add_trace(
        go.Scattergeo(
            lon=node_df["lon"],
            lat=node_df["lat"],
            mode="markers",
            marker=dict(
                size=(node_df["hub_tons"] / node_df["hub_tons"].max() * 24).clip(lower=6),
                color=node_df["hub_tons"],
                colorscale="Blues",
                line=dict(width=0.6, color="white"),
                colorbar=dict(title="Hub tons"),
                opacity=0.85,
            ),
            text=node_df["zone_name"] + ", " + node_df["state_name"],
            hovertemplate="%{text}<br>Hub tons: %{marker.color:,.0f}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        title=f"Top {top_n} DMS Truck Corridors ({year})",
        geo=dict(scope="usa", projection_type="albers usa", showland=True, landcolor="rgb(245,245,245)"),
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.write_html(str(MAP_DIR / f"top_dms_corridors_{year}.html"), include_plotlyjs="cdn")


def main() -> None:
    start = time.time()
    build_dms_corridor_outputs(YEAR, TOP_N)
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()

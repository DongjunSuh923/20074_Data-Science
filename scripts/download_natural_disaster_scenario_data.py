from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import requests


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
OUT_DIR = PROJECT_ROOT / "external_data" / "Scenario_Tier1" / "Natural_Disaster"
STORM_DIR = OUT_DIR / "NOAA_StormEvents"
AQI_DIR = OUT_DIR / "EPA_AQI"
CLIMATE_DIR = OUT_DIR / "NOAA_ClimateAtAGlance"

YEARS = list(range(2018, 2026))
STORM_INDEX_URL = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
EPA_AQI_TEMPLATE = "https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_{year}.zip"
STATE_FIPS = {
    "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO",
    "09": "CT", "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI",
    "16": "ID", "17": "IL", "18": "IN", "19": "IA", "20": "KS", "21": "KY",
    "22": "LA", "23": "ME", "24": "MD", "25": "MA", "26": "MI", "27": "MN",
    "28": "MS", "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", "39": "OH",
    "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA",
    "54": "WV", "55": "WI", "56": "WY",
}


def ensure_dirs() -> None:
    for folder in [STORM_DIR, AQI_DIR, CLIMATE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def download_file(url: str, target: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)


def find_latest_storm_files() -> dict[int, str]:
    response = requests.get(STORM_INDEX_URL, timeout=120)
    response.raise_for_status()
    hrefs = re.findall(r"(StormEvents_details-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz)", response.text)
    latest: dict[int, str] = {}
    for year in YEARS:
        matches = [name for name in hrefs if f"_d{year}_" in name]
        if not matches:
            continue
        latest_name = sorted(matches)[-1]
        latest[year] = STORM_INDEX_URL + latest_name
    return latest


def download_storm_events() -> None:
    latest = find_latest_storm_files()
    for year, url in latest.items():
        target = STORM_DIR / f"stormevents_details_{year}.csv.gz"
        if not target.exists():
            print(f"Downloading storm events {year}")
            download_file(url, target)


def download_aqi() -> None:
    for year in YEARS:
        target = AQI_DIR / f"annual_aqi_by_county_{year}.zip"
        if not target.exists():
            url = EPA_AQI_TEMPLATE.format(year=year)
            print(f"Downloading AQI county {year}")
            download_file(url, target)


def download_climate() -> None:
    metrics = {
        "tavg": "average_temperature",
        "pcp": "precipitation",
    }
    for state_fips, state_abbr in STATE_FIPS.items():
        for metric_code, metric_name in metrics.items():
            target = CLIMATE_DIR / f"{state_abbr}_{metric_name}_2018_2025.csv"
            if target.exists():
                continue
            url = (
                "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/"
                f"statewide/time-series/{state_fips}/{metric_code}/12/12/2018-2025.csv"
                "?base_prd=true&begbaseyear=1901&endbaseyear=2000"
            )
            print(f"Downloading climate {state_abbr} {metric_code}")
            response = requests.get(url, timeout=120)
            if response.status_code != 200:
                print(f"Skipping climate {state_abbr} {metric_code}: HTTP {response.status_code}")
                continue
            target.write_bytes(response.content)


def write_manifest() -> None:
    storm_files = sorted(p.name for p in STORM_DIR.glob("*.csv.gz"))
    aqi_files = sorted(p.name for p in AQI_DIR.glob("*.zip"))
    climate_files = sorted(p.name for p in CLIMATE_DIR.glob("*.csv"))
    manifest = io.StringIO()
    manifest.write("# Natural disaster Tier 1 download manifest\n")
    manifest.write(f"storm_files={len(storm_files)}\n")
    manifest.write(f"aqi_files={len(aqi_files)}\n")
    manifest.write(f"climate_files={len(climate_files)}\n")
    (OUT_DIR / "download_manifest.txt").write_text(manifest.getvalue(), encoding="utf-8")


def validate_aqi_zip_contents() -> None:
    sample = AQI_DIR / "annual_aqi_by_county_2024.zip"
    if not sample.exists():
        return
    with zipfile.ZipFile(sample) as zf:
        names = zf.namelist()
    (OUT_DIR / "aqi_zip_members_2024.txt").write_text("\n".join(names), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    download_storm_events()
    download_aqi()
    download_climate()
    write_manifest()
    validate_aqi_zip_contents()


if __name__ == "__main__":
    main()

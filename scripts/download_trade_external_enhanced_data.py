from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


WORK_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = WORK_ROOT / "external_data" / "Scenario_Tier2" / "Trade_External_Enhanced" / "Census_State_Imports"
YEARS = range(2018, 2026)
MONTHS = range(1, 13)
BASE_URL = "https://www.census.gov/trade/downloads/{year}/state_imp/naics_m/ISNAICS{yy}{month:02d}.ZIP"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        yy = str(year)[-2:]
        for month in MONTHS:
            out_path = OUT_DIR / f"ISNAICS_{year}_{month:02d}.zip"
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            url = BASE_URL.format(year=year, yy=yy, month=month)
            print(f"Downloading {url}")
            urlretrieve(url, out_path)


if __name__ == "__main__":
    main()

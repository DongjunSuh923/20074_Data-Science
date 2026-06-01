from __future__ import annotations

import re
from pathlib import Path

import requests


PROJECT_ROOT = Path(r"C:\Users\서동준\IdeaProjects\FAF5.7.1_2018-2024")
OUT_DIR = PROJECT_ROOT / "external_data" / "Scenario_Tier1" / "Trade_External"
EXPORT_DIR = OUT_DIR / "Census_State_Exports"
INDEX_URL = "https://www.census.gov/foreign-trade/data/STNAICS.html"
YEARS = set(range(2018, 2026))


def ensure_dirs() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_index_links() -> list[str]:
    response = requests.get(INDEX_URL, timeout=120)
    response.raise_for_status()
    links = re.findall(r'href="([^"]+STNAICS\d{4}\.ZIP)"', response.text, flags=re.IGNORECASE)
    normalized: list[str] = []
    for link in links:
        if link.startswith("http"):
            normalized.append(link)
        else:
            normalized.append("https://www.census.gov" + link)
    return normalized


def download_file(url: str, target: Path) -> None:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target.write_bytes(response.content)


def main() -> None:
    ensure_dirs()
    links = fetch_index_links()
    kept: list[tuple[str, str]] = []
    for url in links:
        match = re.search(r"STNAICS(\d{2})(\d{2})\.ZIP", url, flags=re.IGNORECASE)
        if not match:
            continue
        yy = int(match.group(1))
        year = 2000 + yy
        if year not in YEARS:
            continue
        ym = f"{year}-{match.group(2)}"
        target = EXPORT_DIR / f"STNAICS_{year}_{match.group(2)}.zip"
        kept.append((ym, url))
        if target.exists():
            continue
        print(f"Downloading {ym}")
        download_file(url, target)

    manifest_lines = [
        "# Trade / External Network Shock Download Manifest",
        "",
        f"index_url={INDEX_URL}",
        f"download_count={len(kept)}",
    ]
    manifest_lines.extend(f"{ym},{url}" for ym, url in kept)
    (OUT_DIR / "download_manifest.txt").write_text("\n".join(manifest_lines), encoding="utf-8")


if __name__ == "__main__":
    main()

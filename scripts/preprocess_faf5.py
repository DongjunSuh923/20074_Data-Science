from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd


INPUT_CSV = PROJECT_ROOT / "FAF5.7.1_2018-2024.csv"
METADATA_XLSX = PROJECT_ROOT / "FAF5_metadata.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only"
EDA_DIR = OUTPUT_DIR / "eda"
LOOKUP_DIR = OUTPUT_DIR / "lookups"
DB_PATH = OUTPUT_DIR / "faf5_preprocessing.db"
LONG_CSV_PATH = OUTPUT_DIR / "faf5_long_final.csv"
SUMMARY_JSON_PATH = OUTPUT_DIR / "preprocessing_summary.json"
SUMMARY_MD_PATH = OUTPUT_DIR / "preprocessing_summary.md"

YEARS = list(range(2018, 2025))
ID_COLS = [
    "fr_orig",
    "dms_orig",
    "dms_dest",
    "fr_dest",
    "fr_inmode",
    "dms_mode",
    "fr_outmode",
    "sctg2",
    "trade_type",
    "dist_band",
]
NUMERIC_BASES = ["tons", "value", "current_value", "tmiles"]
FINAL_COLS = [
    "fr_orig",
    "dms_orig",
    "dms_dest",
    "fr_dest",
    "fr_inmode",
    "dms_mode",
    "fr_outmode",
    "sctg2",
    "trade_type",
    "dist_band",
    "year",
    "tons",
    "value",
    "current_value",
    "tmiles",
    "tons_zero_flag",
    "value_missing_flag",
    "current_value_missing_flag",
    "tmiles_missing_flag",
    "tons_lag1",
]
TRUCK_MODE = "1"
AIR_INCLUDE_TRUCK_MODE = "4"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    LOOKUP_DIR.mkdir(parents=True, exist_ok=True)


def make_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS faf5_long_staging;
        DROP TABLE IF EXISTS faf5_long_final;

        CREATE TABLE faf5_long_staging (
            fr_orig TEXT,
            dms_orig TEXT,
            dms_dest TEXT,
            fr_dest TEXT,
            fr_inmode TEXT,
            dms_mode TEXT,
            fr_outmode TEXT,
            sctg2 TEXT,
            trade_type TEXT,
            dist_band TEXT,
            year INTEGER,
            tons REAL,
            value REAL,
            current_value REAL,
            tmiles REAL,
            tons_zero_flag INTEGER,
            value_missing_flag INTEGER,
            current_value_missing_flag INTEGER,
            tmiles_missing_flag INTEGER
        );
        """
    )
    conn.commit()


def melt_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        renamed = chunk[
            ID_COLS
            + [
                f"tons_{year}",
                f"value_{year}",
                f"current_value_{year}",
                f"tmiles_{year}",
            ]
        ].copy()
        renamed = renamed.rename(
            columns={
                f"tons_{year}": "tons",
                f"value_{year}": "value",
                f"current_value_{year}": "current_value",
                f"tmiles_{year}": "tmiles",
            }
        )
        renamed["year"] = year
        frames.append(renamed)

    long_df = pd.concat(frames, ignore_index=True)
    for col in NUMERIC_BASES:
        long_df[col] = pd.to_numeric(long_df[col], errors="coerce")

    long_df["tons_zero_flag"] = (long_df["tons"] == 0).astype("int8")
    long_df["value_missing_flag"] = long_df["value"].isna().astype("int8")
    long_df["current_value_missing_flag"] = long_df["current_value"].isna().astype("int8")
    long_df["tmiles_missing_flag"] = long_df["tmiles"].isna().astype("int8")

    return long_df[
        [
            "fr_orig",
            "dms_orig",
            "dms_dest",
            "fr_dest",
            "fr_inmode",
            "dms_mode",
            "fr_outmode",
            "sctg2",
            "trade_type",
            "dist_band",
            "year",
            "tons",
            "value",
            "current_value",
            "tmiles",
            "tons_zero_flag",
            "value_missing_flag",
            "current_value_missing_flag",
            "tmiles_missing_flag",
        ]
    ]


def summarize_distribution(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(clean.shape[0]),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "p05": float(clean.quantile(0.05)),
        "p25": float(clean.quantile(0.25)),
        "p75": float(clean.quantile(0.75)),
        "p95": float(clean.quantile(0.95)),
        "max": float(clean.max()),
    }


def load_metadata_lookups() -> dict[str, pd.DataFrame]:
    wb = openpyxl.load_workbook(METADATA_XLSX, read_only=True, data_only=True)

    def sheet_rows(sheet_name: str, columns: list[str], start_row: int = 2) -> pd.DataFrame:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(min_row=start_row, values_only=True):
            if not row or row[0] is None:
                continue
            rows.append(row[: len(columns)])
        return pd.DataFrame(rows, columns=columns)

    state = sheet_rows("State", ["state_fips", "state_name"]).astype(str)
    state["state_fips"] = state["state_fips"].str.zfill(2)

    domestic = sheet_rows(
        "FAF Zone (Domestic)",
        ["dms_code", "dms_short_desc", "dms_long_desc"],
    ).astype(str)
    domestic["dms_code"] = domestic["dms_code"].str.zfill(3)
    domestic["state_fips"] = domestic["dms_code"].str[:2]
    domestic = domestic.merge(state, on="state_fips", how="left")

    foreign = sheet_rows("FAF Zone (Foreign)", ["fr_code", "fr_desc"]).astype(str)
    commodity = sheet_rows("Commodity (SCTG2)", ["sctg2", "sctg2_desc"]).astype(str)
    mode = sheet_rows("Mode", ["mode_code", "mode_desc"]).astype(str)
    trade_type = sheet_rows("Trade Type", ["trade_type", "trade_type_desc"]).astype(str)
    distance_band = sheet_rows("Distance Band", ["dist_band", "dist_band_desc"]).astype(str)

    for df in [state, domestic, foreign, commodity, mode, trade_type, distance_band]:
        for col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("_x000D_", "", regex=False)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )

    return {
        "state": state,
        "domestic": domestic,
        "foreign": foreign,
        "commodity": commodity,
        "mode": mode,
        "trade_type": trade_type,
        "distance_band": distance_band,
    }


def export_lookup_tables(lookups: dict[str, pd.DataFrame]) -> None:
    lookups["state"].to_csv(LOOKUP_DIR / "state_lookup.csv", index=False)
    lookups["domestic"].to_csv(LOOKUP_DIR / "domestic_zone_lookup.csv", index=False)
    lookups["foreign"].to_csv(LOOKUP_DIR / "foreign_zone_lookup.csv", index=False)
    lookups["commodity"].to_csv(LOOKUP_DIR / "commodity_lookup.csv", index=False)
    lookups["mode"].to_csv(LOOKUP_DIR / "mode_lookup.csv", index=False)
    lookups["trade_type"].to_csv(LOOKUP_DIR / "trade_type_lookup.csv", index=False)
    lookups["distance_band"].to_csv(LOOKUP_DIR / "distance_band_lookup.csv", index=False)


def build_map_ready_route_summary(
    conn: sqlite3.Connection,
    lookups: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    route_yearly = pd.read_sql_query(
        """
        SELECT
            dms_orig,
            dms_dest,
            year,
            SUM(tons) AS total_tons,
            SUM(current_value) AS total_current_value,
            SUM(value) AS total_value,
            SUM(tmiles) AS total_tmiles
        FROM faf5_long_final
        GROUP BY dms_orig, dms_dest, year
        ORDER BY dms_orig, dms_dest, year;
        """,
        conn,
    )

    domestic = lookups["domestic"].rename(
        columns={
            "dms_code": "dms_orig",
            "dms_short_desc": "orig_zone_name",
            "dms_long_desc": "orig_zone_long_name",
            "state_fips": "orig_state_fips",
            "state_name": "orig_state_name",
        }
    )
    route_yearly = route_yearly.merge(domestic, on="dms_orig", how="left")

    domestic_dest = lookups["domestic"].rename(
        columns={
            "dms_code": "dms_dest",
            "dms_short_desc": "dest_zone_name",
            "dms_long_desc": "dest_zone_long_name",
            "state_fips": "dest_state_fips",
            "state_name": "dest_state_name",
        }
    )
    route_yearly = route_yearly.merge(domestic_dest, on="dms_dest", how="left")
    route_yearly.to_csv(OUTPUT_DIR / "route_yearly_summary_for_map.csv", index=False)
    return route_yearly


def preprocess() -> dict:
    ensure_dirs()
    conn = make_connection()
    initialize_database(conn)

    dtype_map = {col: "string" for col in ID_COLS}
    chunksize = 100_000

    summary = {
        "source_csv": str(INPUT_CSV),
        "generated_at_epoch": int(time.time()),
        "chunksize": chunksize,
        "years": YEARS,
        "wide_rows_read": 0,
        "long_rows_generated": 0,
        "tons_missing_rows": 0,
        "value_missing_rows": 0,
        "current_value_missing_rows": 0,
        "tmiles_missing_rows": 0,
        "tons_negative_rows": 0,
        "tmiles_negative_rows": 0,
        "tons_zero_rows": 0,
        "truck_mode_filter": TRUCK_MODE,
        "pre_filter_mode_counts": {},
    }

    reader = pd.read_csv(INPUT_CSV, dtype=dtype_map, chunksize=chunksize, low_memory=False)
    for idx, chunk in enumerate(reader, start=1):
        summary["wide_rows_read"] += int(chunk.shape[0])
        mode_counts = chunk["dms_mode"].fillna("MISSING").astype(str).value_counts().to_dict()
        for mode, count in mode_counts.items():
            summary["pre_filter_mode_counts"][mode] = summary["pre_filter_mode_counts"].get(mode, 0) + int(count)

        chunk = chunk.loc[chunk["dms_mode"] == TRUCK_MODE].copy()
        if chunk.empty:
            print(f"[chunk {idx}] no truck rows after dms_mode filter")
            continue

        long_df = melt_chunk(chunk)
        summary["long_rows_generated"] += int(long_df.shape[0])
        summary["tons_missing_rows"] += int(long_df["tons"].isna().sum())
        summary["value_missing_rows"] += int(long_df["value"].isna().sum())
        summary["current_value_missing_rows"] += int(long_df["current_value"].isna().sum())
        summary["tmiles_missing_rows"] += int(long_df["tmiles"].isna().sum())
        summary["tons_negative_rows"] += int((long_df["tons"] < 0).sum())
        summary["tmiles_negative_rows"] += int((long_df["tmiles"] < 0).sum())
        summary["tons_zero_rows"] += int((long_df["tons"] == 0).sum())

        cleaned = long_df.loc[
            long_df["tons"].notna()
            & (long_df["tons"] >= 0)
            & (long_df["tmiles"].isna() | (long_df["tmiles"] >= 0))
        ].copy()

        cleaned.to_sql("faf5_long_staging", conn, if_exists="append", index=False)
        print(
            f"[chunk {idx}] wide_rows={chunk.shape[0]:,} "
            f"long_rows={long_df.shape[0]:,} kept_rows={cleaned.shape[0]:,}"
        )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_faf5_group_year
        ON faf5_long_staging (fr_orig, dms_dest, sctg2, year);
        """
    )
    conn.commit()

    conn.execute("DROP TABLE IF EXISTS faf5_long_final;")
    conn.execute(
        """
        CREATE TABLE faf5_long_final AS
        SELECT
            fr_orig,
            dms_orig,
            dms_dest,
            fr_dest,
            fr_inmode,
            dms_mode,
            fr_outmode,
            sctg2,
            trade_type,
            dist_band,
            year,
            tons,
            value,
            current_value,
            tmiles,
            tons_zero_flag,
            value_missing_flag,
            current_value_missing_flag,
            tmiles_missing_flag,
            LAG(tons) OVER (
                PARTITION BY fr_orig, dms_dest, sctg2
                ORDER BY year
            ) AS tons_lag1
        FROM faf5_long_staging
        ORDER BY fr_orig, dms_dest, sctg2, year;
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_faf5_final_year
        ON faf5_long_final (year);
        """
    )
    conn.commit()

    summary["final_long_rows"] = int(
        conn.execute("SELECT COUNT(*) FROM faf5_long_final;").fetchone()[0]
    )
    summary["truck_wide_rows"] = int(summary["final_long_rows"] / len(YEARS))
    summary["distinct_group_count"] = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM (
                SELECT fr_orig, dms_dest, sctg2
                FROM faf5_long_final
                GROUP BY fr_orig, dms_dest, sctg2
            );
            """
        ).fetchone()[0]
    )

    return summary


def export_final_csv(conn: sqlite3.Connection) -> None:
    target_path = LONG_CSV_PATH
    if target_path.exists():
        try:
            target_path.unlink()
        except PermissionError:
            target_path = target_path.with_name(f"{target_path.stem}_refreshed{target_path.suffix}")

    query = f"SELECT {', '.join(FINAL_COLS)} FROM faf5_long_final"
    for idx, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=200_000), start=1):
        chunk.to_csv(target_path, mode="a", index=False, header=(idx == 1))
        print(f"[export {idx}] wrote_rows={chunk.shape[0]:,}")


def save_query_csv(conn: sqlite3.Connection, query: str, path: Path) -> pd.DataFrame:
    df = pd.read_sql_query(query, conn)
    df.to_csv(path, index=False)
    return df


def make_eda(conn: sqlite3.Connection, lookups: dict[str, pd.DataFrame]) -> dict:
    plt.style.use("ggplot")
    eda_summary: dict[str, dict] = {}

    yearly = save_query_csv(
        conn,
        """
        SELECT year, SUM(tons) AS total_tons, SUM(current_value) AS total_current_value
        FROM faf5_long_final
        GROUP BY year
        ORDER BY year;
        """,
        EDA_DIR / "yearly_totals.csv",
    )
    plt.figure(figsize=(9, 5))
    plt.plot(yearly["year"], yearly["total_tons"], marker="o", linewidth=2.5, color="#1f77b4")
    plt.title("Yearly Total Tons")
    plt.xlabel("Year")
    plt.ylabel("Sum of Tons")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "yearly_total_tons.png", dpi=180)
    plt.close()
    eda_summary["yearly"] = yearly.to_dict(orient="records")

    sctg2 = save_query_csv(
        conn,
        """
        SELECT sctg2, SUM(tons) AS total_tons
        FROM faf5_long_final
        GROUP BY sctg2
        ORDER BY total_tons DESC;
        """,
        EDA_DIR / "tons_by_sctg2.csv",
    )
    top_sctg2 = sctg2.head(20).iloc[::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(top_sctg2["sctg2"].astype(str), top_sctg2["total_tons"], color="#2a9d8f")
    plt.title("Top 20 Commodity Groups by Tons")
    plt.xlabel("Sum of Tons")
    plt.ylabel("sctg2")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "top_sctg2_tons.png", dpi=180)
    plt.close()
    eda_summary["top_sctg2"] = sctg2.head(10).to_dict(orient="records")

    dist_band = save_query_csv(
        conn,
        """
        SELECT dist_band, SUM(tons) AS total_tons
        FROM faf5_long_final
        GROUP BY dist_band
        ORDER BY CAST(dist_band AS INTEGER);
        """,
        EDA_DIR / "tons_by_dist_band.csv",
    )
    plt.figure(figsize=(9, 5))
    plt.bar(dist_band["dist_band"].astype(str), dist_band["total_tons"], color="#e76f51")
    plt.title("Tons by Distance Band")
    plt.xlabel("dist_band")
    plt.ylabel("Sum of Tons")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "tons_by_dist_band.png", dpi=180)
    plt.close()
    eda_summary["dist_band"] = dist_band.to_dict(orient="records")

    top_routes = save_query_csv(
        conn,
        """
        SELECT
            fr_orig,
            dms_orig,
            dms_dest,
            fr_dest,
            SUM(tons) AS total_tons
        FROM faf5_long_final
        GROUP BY fr_orig, dms_orig, dms_dest, fr_dest
        ORDER BY total_tons DESC
        LIMIT 20;
        """,
        EDA_DIR / "top_routes.csv",
    )
    top_routes = top_routes.merge(
        lookups["domestic"].rename(
            columns={
                "dms_code": "dms_orig",
                "dms_short_desc": "orig_zone_name",
                "dms_long_desc": "orig_zone_long_name",
                "state_fips": "orig_state_fips",
                "state_name": "orig_state_name",
            }
        ),
        on="dms_orig",
        how="left",
    ).merge(
        lookups["domestic"].rename(
            columns={
                "dms_code": "dms_dest",
                "dms_short_desc": "dest_zone_name",
                "dms_long_desc": "dest_zone_long_name",
                "state_fips": "dest_state_fips",
                "state_name": "dest_state_name",
            }
        ),
        on="dms_dest",
        how="left",
    )
    top_routes.to_csv(EDA_DIR / "top_routes.csv", index=False)
    route_labels = [
        f"{row['orig_zone_name']} -> {row['dest_zone_name']}"
        for _, row in top_routes.iloc[::-1].iterrows()
    ]
    plt.figure(figsize=(12, 8))
    plt.barh(route_labels, top_routes["total_tons"].iloc[::-1], color="#264653")
    plt.title("Top 20 Routes by Tons")
    plt.xlabel("Sum of Tons")
    plt.ylabel("Route")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "top_routes.png", dpi=180)
    plt.close()
    eda_summary["top_routes"] = top_routes.to_dict(orient="records")

    scatter = pd.read_sql_query(
        """
        SELECT tons, current_value
        FROM faf5_long_final
        WHERE tons IS NOT NULL
          AND current_value IS NOT NULL
          AND tons >= 0
          AND current_value >= 0
          AND rowid % 250 = 0
        LIMIT 50000;
        """,
        conn,
    )
    plt.figure(figsize=(8, 6))
    plt.scatter(
        scatter["tons"],
        scatter["current_value"],
        s=8,
        alpha=0.25,
        color="#6a4c93",
        edgecolors="none",
    )
    plt.title("Tons vs Current Value")
    plt.xlabel("Tons")
    plt.ylabel("Current Value")
    plt.tight_layout()
    plt.savefig(EDA_DIR / "tons_vs_current_value.png", dpi=180)
    plt.close()
    eda_summary["scatter_sample_size"] = int(scatter.shape[0])

    dist_query = {
        "tons": "SELECT tons AS metric FROM faf5_long_final WHERE tons IS NOT NULL AND tons >= 0;",
        "value": "SELECT value AS metric FROM faf5_long_final WHERE value IS NOT NULL AND value >= 0;",
        "tmiles": "SELECT tmiles AS metric FROM faf5_long_final WHERE tmiles IS NOT NULL AND tmiles >= 0;",
    }
    dist_summary = {}
    for metric, query in dist_query.items():
        sample = pd.read_sql_query(
            query.replace(";", " AND rowid % 200 = 0 LIMIT 200000;"),
            conn,
        )
        if metric == "tons":
            plot_values = sample["metric"].clip(lower=1e-9)
            plt.figure(figsize=(8, 5))
            plt.hist(plot_values, bins=60, color="#457b9d", alpha=0.9)
            plt.xscale("log")
            plt.title("Distribution of Tons (log scale)")
            plt.xlabel("Tons")
            plt.ylabel("Frequency")
        else:
            plt.figure(figsize=(8, 5))
            plt.hist(sample["metric"], bins=60, color="#f4a261", alpha=0.9)
            plt.title(f"Distribution of {metric.title()}")
            plt.xlabel(metric)
            plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"distribution_{metric}.png", dpi=180)
        plt.close()
        dist_summary[metric] = summarize_distribution(sample["metric"])

    eda_summary["distribution_summary"] = dist_summary
    return eda_summary


def save_mode_comparison(summary: dict) -> list[dict]:
    mode_map = {
        "1": "Truck",
        "2": "Rail",
        "3": "Water",
        "4": "Air (include truck-air)",
        "5": "Multiple modes & mail",
        "6": "Pipeline",
        "7": "Other and unknown",
        "8": "No domestic mode",
    }
    records = []
    total = sum(summary["pre_filter_mode_counts"].values())
    for mode in ["1", "4"]:
        count = int(summary["pre_filter_mode_counts"].get(mode, 0))
        records.append(
            {
                "dms_mode": mode,
                "description": mode_map.get(mode, "Unknown"),
                "wide_row_count": count,
                "share_of_wide_rows": (count / total) if total else None,
            }
        )
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "mode_1_vs_4_comparison.csv", index=False)
    return records


def write_markdown_summary(summary: dict, eda_summary: dict) -> None:
    lines = [
        "# FAF5 Preprocessing Summary",
        "",
        "## Output Files",
        f"- Long format CSV: `{LONG_CSV_PATH}`",
        f"- SQLite database: `{DB_PATH}`",
        f"- EDA directory: `{EDA_DIR}`",
        f"- Lookup directory: `{LOOKUP_DIR}`",
        f"- Map-ready route summary: `{OUTPUT_DIR / 'route_yearly_summary_for_map.csv'}`",
        "",
        "## Row Counts",
        f"- Wide rows read: {summary['wide_rows_read']:,}",
        f"- Wide rows kept after `dms_mode == {summary['truck_mode_filter']}`: {summary['truck_wide_rows']:,}",
        f"- Long rows generated: {summary['long_rows_generated']:,}",
        f"- Final long rows kept: {summary['final_long_rows']:,}",
        f"- Distinct `(fr_orig, dms_dest, sctg2)` groups: {summary['distinct_group_count']:,}",
        "",
        "## Missing / Outlier Checks",
        f"- Missing `tons`: {summary['tons_missing_rows']:,}",
        f"- Missing `value`: {summary['value_missing_rows']:,}",
        f"- Missing `current_value`: {summary['current_value_missing_rows']:,}",
        f"- Missing `tmiles`: {summary['tmiles_missing_rows']:,}",
        f"- Negative `tons` removed: {summary['tons_negative_rows']:,}",
        f"- Negative `tmiles` removed: {summary['tmiles_negative_rows']:,}",
        f"- Zero `tons` flagged: {summary['tons_zero_rows']:,}",
        "",
        "## Notes",
        f"- This run keeps only `dms_mode == {summary['truck_mode_filter']}` (`Truck`) to match the scenario PDF.",
        "- `dms_orig` and `dms_dest` should be treated as the primary geography for US truck network analysis.",
        "- `fr_orig` and `fr_dest` are foreign-region columns, so they are structurally blank for domestic flows.",
        "- Rows with missing `tons` were removed.",
        "- Rows with negative `tons` or negative `tmiles` were removed.",
        "- Missing `value`, `current_value`, and `tmiles` were retained and flagged for later feature decisions.",
        "- `tons_lag1` was generated after sorting by `(fr_orig, dms_dest, sctg2, year)`.",
        "",
        "## EDA Highlights",
        f"- Yearly points generated: {len(eda_summary['yearly'])}",
        f"- Commodity categories summarized: {len(eda_summary['top_sctg2'])} shown in summary, full CSV saved",
        f"- Route ranking rows saved: {len(eda_summary['top_routes'])}",
        f"- Scatter sample size: {eda_summary['scatter_sample_size']:,}",
    ]
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = preprocess()
    conn = make_connection()
    lookups = load_metadata_lookups()
    export_lookup_tables(lookups)
    export_final_csv(conn)
    route_yearly = build_map_ready_route_summary(conn, lookups)
    eda_summary = make_eda(conn, lookups)
    summary["mode_1_vs_4_comparison"] = save_mode_comparison(summary)
    summary["map_ready_route_rows"] = int(route_yearly.shape[0])
    summary["eda"] = eda_summary
    SUMMARY_JSON_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_summary(summary, eda_summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

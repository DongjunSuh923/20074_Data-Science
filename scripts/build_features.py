from __future__ import annotations

import json
import math
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
import pandas as pd


PREPROCESS_DIR = PROJECT_ROOT / "outputs" / "preprocessing_truck_only"
INPUT_DB = PREPROCESS_DIR / "faf5_preprocessing.db"
LOOKUP_DIR = PREPROCESS_DIR / "lookups"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
EDA_DIR = OUTPUT_DIR / "validation"
FEATURE_DB = OUTPUT_DIR / "faf5_features.db"
FEATURE_ALL_CSV = OUTPUT_DIR / "feature_dataset_all_rows.csv"
FEATURE_MODEL_CSV = OUTPUT_DIR / "feature_dataset_model_ready.csv"
FEATURE_2025_CSV = OUTPUT_DIR / "feature_dataset_forecast_2025.csv"
FEATURE_DOC_MD = OUTPUT_DIR / "feature_description.md"
SUMMARY_JSON = OUTPUT_DIR / "feature_summary.json"
SUMMARY_MD = OUTPUT_DIR / "feature_summary.md"

KEY_COLS = [
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

EXPORT_COLS = [
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
    "orig_state_fips",
    "dest_state_fips",
    "year",
    "year_index",
    "target_tons",
    "target_log_tons",
    "tons_lag1",
    "tons_lag2",
    "tons_growth_rate",
    "tons_diff1",
    "current_value_lag1",
    "tmiles_lag1",
    "value_per_ton_lag1",
    "tons_per_tmile_lag1",
    "orig_total_outbound_tons_lag1",
    "dest_total_inbound_tons_lag1",
    "orig_unique_dest_count_lag1",
    "dest_unique_orig_count_lag1",
    "orig_unique_sctg2_count_lag1",
    "dest_unique_sctg2_count_lag1",
    "route_total_tons_lag1",
    "route_total_tons_lag2",
    "route_growth_rate",
    "route_share_of_origin_lag1",
    "route_share_of_dest_lag1",
    "corridor_trade_type_count_lag1",
    "corridor_sctg2_count_lag1",
    "tons_zero_flag_lag1",
    "is_domestic_trade",
    "is_import_trade",
    "is_export_trade",
    "feature_group_history",
    "feature_group_cargo",
    "feature_group_value_efficiency",
    "feature_group_time",
    "feature_group_network",
]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EDA_DIR.mkdir(parents=True, exist_ok=True)


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-200000;")
    return conn


def setup_feature_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS feature_all_rows;
        DROP TABLE IF EXISTS feature_model_ready;
        DROP TABLE IF EXISTS feature_forecast_2025;
        """
    )
    conn.commit()


def build_feature_table(input_conn: sqlite3.Connection, feature_conn: sqlite3.Connection) -> None:
    feature_conn.execute("ATTACH DATABASE ? AS src", (str(INPUT_DB),))
    feature_conn.executescript(
        """
        DROP TABLE IF EXISTS feature_all_rows;

        CREATE TABLE feature_all_rows AS
        WITH base AS (
            SELECT
                COALESCE(fr_orig, 'MISSING') AS fr_orig_key,
                dms_orig,
                dms_dest,
                COALESCE(fr_dest, 'MISSING') AS fr_dest_key,
                COALESCE(fr_inmode, 'MISSING') AS fr_inmode_key,
                dms_mode,
                COALESCE(fr_outmode, 'MISSING') AS fr_outmode_key,
                sctg2,
                trade_type,
                dist_band,
                fr_orig,
                fr_dest,
                fr_inmode,
                fr_outmode,
                year,
                tons,
                current_value,
                tmiles,
                tons_zero_flag
            FROM src.faf5_long_final
        ),
        row_level AS (
            SELECT
                *,
                LAG(tons, 1) OVER (
                    PARTITION BY fr_orig_key, dms_orig, dms_dest, fr_dest_key, fr_inmode_key, dms_mode, fr_outmode_key, sctg2, trade_type, dist_band
                    ORDER BY year
                ) AS tons_lag1,
                LAG(tons, 2) OVER (
                    PARTITION BY fr_orig_key, dms_orig, dms_dest, fr_dest_key, fr_inmode_key, dms_mode, fr_outmode_key, sctg2, trade_type, dist_band
                    ORDER BY year
                ) AS tons_lag2,
                LAG(current_value, 1) OVER (
                    PARTITION BY fr_orig_key, dms_orig, dms_dest, fr_dest_key, fr_inmode_key, dms_mode, fr_outmode_key, sctg2, trade_type, dist_band
                    ORDER BY year
                ) AS current_value_lag1,
                LAG(tmiles, 1) OVER (
                    PARTITION BY fr_orig_key, dms_orig, dms_dest, fr_dest_key, fr_inmode_key, dms_mode, fr_outmode_key, sctg2, trade_type, dist_band
                    ORDER BY year
                ) AS tmiles_lag1,
                LAG(tons_zero_flag, 1) OVER (
                    PARTITION BY fr_orig_key, dms_orig, dms_dest, fr_dest_key, fr_inmode_key, dms_mode, fr_outmode_key, sctg2, trade_type, dist_band
                    ORDER BY year
                ) AS tons_zero_flag_lag1
            FROM base
        ),
        orig_year AS (
            SELECT
                dms_orig,
                year,
                SUM(tons) AS orig_total_outbound_tons,
                COUNT(DISTINCT dms_dest) AS orig_unique_dest_count,
                COUNT(DISTINCT sctg2) AS orig_unique_sctg2_count
            FROM base
            GROUP BY dms_orig, year
        ),
        dest_year AS (
            SELECT
                dms_dest,
                year,
                SUM(tons) AS dest_total_inbound_tons,
                COUNT(DISTINCT dms_orig) AS dest_unique_orig_count,
                COUNT(DISTINCT sctg2) AS dest_unique_sctg2_count
            FROM base
            GROUP BY dms_dest, year
        ),
        route_year AS (
            SELECT
                dms_orig,
                dms_dest,
                year,
                SUM(tons) AS route_total_tons,
                COUNT(DISTINCT trade_type) AS corridor_trade_type_count,
                COUNT(DISTINCT sctg2) AS corridor_sctg2_count
            FROM base
            GROUP BY dms_orig, dms_dest, year
        ),
        route_year_lagged AS (
            SELECT
                dms_orig,
                dms_dest,
                year + 1 AS year,
                route_total_tons AS route_total_tons_lag1,
                LAG(route_total_tons, 1) OVER (
                    PARTITION BY dms_orig, dms_dest ORDER BY year
                ) AS route_total_tons_lag2,
                corridor_trade_type_count AS corridor_trade_type_count_lag1,
                corridor_sctg2_count AS corridor_sctg2_count_lag1
            FROM route_year
        )
        SELECT
            r.fr_orig,
            r.dms_orig,
            r.dms_dest,
            r.fr_dest,
            r.fr_inmode,
            r.dms_mode,
            r.fr_outmode,
            r.sctg2,
            r.trade_type,
            r.dist_band,
            substr(printf('%03d', r.dms_orig), 1, 2) AS orig_state_fips,
            substr(printf('%03d', r.dms_dest), 1, 2) AS dest_state_fips,
            r.year,
            r.year - 2018 AS year_index,
            r.tons AS target_tons,
            ln(1 + r.tons) AS target_log_tons,
            r.tons_lag1,
            r.tons_lag2,
            CASE
                WHEN r.tons_lag2 IS NULL OR r.tons_lag2 = 0 THEN NULL
                ELSE (r.tons_lag1 - r.tons_lag2) / ABS(r.tons_lag2)
            END AS tons_growth_rate,
            CASE
                WHEN r.tons_lag1 IS NULL OR r.tons_lag2 IS NULL THEN NULL
                ELSE r.tons_lag1 - r.tons_lag2
            END AS tons_diff1,
            r.current_value_lag1,
            r.tmiles_lag1,
            CASE
                WHEN r.tons_lag1 IS NULL OR r.tons_lag1 = 0 OR r.current_value_lag1 IS NULL THEN NULL
                ELSE r.current_value_lag1 / r.tons_lag1
            END AS value_per_ton_lag1,
            CASE
                WHEN r.tmiles_lag1 IS NULL OR r.tmiles_lag1 = 0 OR r.tons_lag1 IS NULL THEN NULL
                ELSE r.tons_lag1 / r.tmiles_lag1
            END AS tons_per_tmile_lag1,
            o.orig_total_outbound_tons AS orig_total_outbound_tons_lag1,
            d.dest_total_inbound_tons AS dest_total_inbound_tons_lag1,
            o.orig_unique_dest_count AS orig_unique_dest_count_lag1,
            d.dest_unique_orig_count AS dest_unique_orig_count_lag1,
            o.orig_unique_sctg2_count AS orig_unique_sctg2_count_lag1,
            d.dest_unique_sctg2_count AS dest_unique_sctg2_count_lag1,
            ry.route_total_tons_lag1,
            ry.route_total_tons_lag2,
            CASE
                WHEN ry.route_total_tons_lag2 IS NULL OR ry.route_total_tons_lag2 = 0 THEN NULL
                ELSE (ry.route_total_tons_lag1 - ry.route_total_tons_lag2) / ABS(ry.route_total_tons_lag2)
            END AS route_growth_rate,
            CASE
                WHEN o.orig_total_outbound_tons IS NULL OR o.orig_total_outbound_tons = 0 OR ry.route_total_tons_lag1 IS NULL THEN NULL
                ELSE ry.route_total_tons_lag1 / o.orig_total_outbound_tons
            END AS route_share_of_origin_lag1,
            CASE
                WHEN d.dest_total_inbound_tons IS NULL OR d.dest_total_inbound_tons = 0 OR ry.route_total_tons_lag1 IS NULL THEN NULL
                ELSE ry.route_total_tons_lag1 / d.dest_total_inbound_tons
            END AS route_share_of_dest_lag1,
            ry.corridor_trade_type_count_lag1,
            ry.corridor_sctg2_count_lag1,
            r.tons_zero_flag_lag1,
            CASE WHEN r.trade_type = 1 THEN 1 ELSE 0 END AS is_domestic_trade,
            CASE WHEN r.trade_type = 2 THEN 1 ELSE 0 END AS is_import_trade,
            CASE WHEN r.trade_type = 3 THEN 1 ELSE 0 END AS is_export_trade,
            1 AS feature_group_history,
            1 AS feature_group_cargo,
            1 AS feature_group_value_efficiency,
            1 AS feature_group_time,
            1 AS feature_group_network
        FROM row_level r
        LEFT JOIN orig_year o
            ON r.dms_orig = o.dms_orig
           AND r.year - 1 = o.year
        LEFT JOIN dest_year d
            ON r.dms_dest = d.dms_dest
           AND r.year - 1 = d.year
        LEFT JOIN route_year_lagged ry
            ON r.dms_orig = ry.dms_orig
           AND r.dms_dest = ry.dms_dest
           AND r.year = ry.year
        ORDER BY r.year, r.dms_orig, r.dms_dest, r.sctg2;

        CREATE INDEX IF NOT EXISTS idx_feature_all_year ON feature_all_rows (year);
        CREATE INDEX IF NOT EXISTS idx_feature_all_route ON feature_all_rows (dms_orig, dms_dest, sctg2, trade_type, year);
        """
    )
    feature_conn.commit()
    feature_conn.execute("DETACH DATABASE src")


def build_model_ready_and_forecast(feature_conn: sqlite3.Connection) -> None:
    feature_conn.execute("ATTACH DATABASE ? AS src", (str(INPUT_DB),))
    feature_conn.executescript(
        """
        DROP TABLE IF EXISTS feature_model_ready;
        CREATE TABLE feature_model_ready AS
        SELECT *
        FROM feature_all_rows
        WHERE tons_lag1 IS NOT NULL
          AND tons_lag2 IS NOT NULL
          AND year >= 2020;

        DROP TABLE IF EXISTS feature_forecast_2025;
        CREATE TABLE feature_forecast_2025 AS
        WITH base_2024 AS (
            SELECT
                f.fr_orig,
                f.dms_orig,
                f.dms_dest,
                f.fr_dest,
                f.fr_inmode,
                f.dms_mode,
                f.fr_outmode,
                f.sctg2,
                f.trade_type,
                f.dist_band,
                f.orig_state_fips,
                f.dest_state_fips,
                s.tons AS tons_lag1,
                f.tons_lag1 AS tons_lag2,
                s.current_value AS current_value_lag1,
                s.tmiles AS tmiles_lag1,
                CASE WHEN s.tons = 0 THEN NULL ELSE s.current_value / s.tons END AS value_per_ton_lag1,
                CASE WHEN s.tmiles = 0 THEN NULL ELSE s.tons / s.tmiles END AS tons_per_tmile_lag1,
                s.tons_zero_flag AS tons_zero_flag_lag1,
                f.is_domestic_trade,
                f.is_import_trade,
                f.is_export_trade
            FROM feature_all_rows f
            JOIN src.faf5_long_final s
              ON COALESCE(f.fr_orig, 'MISSING') = COALESCE(s.fr_orig, 'MISSING')
             AND f.dms_orig = s.dms_orig
             AND f.dms_dest = s.dms_dest
             AND COALESCE(f.fr_dest, 'MISSING') = COALESCE(s.fr_dest, 'MISSING')
             AND COALESCE(f.fr_inmode, 'MISSING') = COALESCE(s.fr_inmode, 'MISSING')
             AND f.dms_mode = s.dms_mode
             AND COALESCE(f.fr_outmode, 'MISSING') = COALESCE(s.fr_outmode, 'MISSING')
             AND f.sctg2 = s.sctg2
             AND f.trade_type = s.trade_type
             AND f.dist_band = s.dist_band
             AND s.year = 2024
            WHERE f.year = 2024
        ),
        orig_2024 AS (
            SELECT dms_orig, SUM(tons) AS orig_total_outbound_tons_lag1,
                   COUNT(DISTINCT dms_dest) AS orig_unique_dest_count_lag1,
                   COUNT(DISTINCT sctg2) AS orig_unique_sctg2_count_lag1
            FROM src.faf5_long_final
            WHERE year = 2024
            GROUP BY dms_orig
        ),
        dest_2024 AS (
            SELECT dms_dest, SUM(tons) AS dest_total_inbound_tons_lag1,
                   COUNT(DISTINCT dms_orig) AS dest_unique_orig_count_lag1,
                   COUNT(DISTINCT sctg2) AS dest_unique_sctg2_count_lag1
            FROM src.faf5_long_final
            WHERE year = 2024
            GROUP BY dms_dest
        ),
        route_2024 AS (
            SELECT dms_orig, dms_dest,
                   SUM(tons) AS route_total_tons_lag1,
                   COUNT(DISTINCT trade_type) AS corridor_trade_type_count_lag1,
                   COUNT(DISTINCT sctg2) AS corridor_sctg2_count_lag1
            FROM src.faf5_long_final
            WHERE year = 2024
            GROUP BY dms_orig, dms_dest
        ),
        route_2023 AS (
            SELECT dms_orig, dms_dest, SUM(tons) AS route_total_tons_lag2
            FROM src.faf5_long_final
            WHERE year = 2023
            GROUP BY dms_orig, dms_dest
        )
        SELECT
            b.fr_orig,
            b.dms_orig,
            b.dms_dest,
            b.fr_dest,
            b.fr_inmode,
            b.dms_mode,
            b.fr_outmode,
            b.sctg2,
            b.trade_type,
            b.dist_band,
            b.orig_state_fips,
            b.dest_state_fips,
            2025 AS year,
            7 AS year_index,
            NULL AS target_tons,
            NULL AS target_log_tons,
            b.tons_lag1,
            b.tons_lag2,
            CASE WHEN b.tons_lag2 IS NULL OR b.tons_lag2 = 0 THEN NULL ELSE (b.tons_lag1 - b.tons_lag2) / ABS(b.tons_lag2) END AS tons_growth_rate,
            CASE WHEN b.tons_lag1 IS NULL OR b.tons_lag2 IS NULL THEN NULL ELSE b.tons_lag1 - b.tons_lag2 END AS tons_diff1,
            b.current_value_lag1,
            b.tmiles_lag1,
            b.value_per_ton_lag1,
            b.tons_per_tmile_lag1,
            o.orig_total_outbound_tons_lag1,
            d.dest_total_inbound_tons_lag1,
            o.orig_unique_dest_count_lag1,
            d.dest_unique_orig_count_lag1,
            o.orig_unique_sctg2_count_lag1,
            d.dest_unique_sctg2_count_lag1,
            r24.route_total_tons_lag1,
            r23.route_total_tons_lag2,
            CASE WHEN r23.route_total_tons_lag2 IS NULL OR r23.route_total_tons_lag2 = 0 THEN NULL ELSE (r24.route_total_tons_lag1 - r23.route_total_tons_lag2) / ABS(r23.route_total_tons_lag2) END AS route_growth_rate,
            CASE WHEN o.orig_total_outbound_tons_lag1 IS NULL OR o.orig_total_outbound_tons_lag1 = 0 THEN NULL ELSE r24.route_total_tons_lag1 / o.orig_total_outbound_tons_lag1 END AS route_share_of_origin_lag1,
            CASE WHEN d.dest_total_inbound_tons_lag1 IS NULL OR d.dest_total_inbound_tons_lag1 = 0 THEN NULL ELSE r24.route_total_tons_lag1 / d.dest_total_inbound_tons_lag1 END AS route_share_of_dest_lag1,
            r24.corridor_trade_type_count_lag1,
            r24.corridor_sctg2_count_lag1,
            b.tons_zero_flag_lag1,
            b.is_domestic_trade,
            b.is_import_trade,
            b.is_export_trade,
            1 AS feature_group_history,
            1 AS feature_group_cargo,
            1 AS feature_group_value_efficiency,
            1 AS feature_group_time,
            1 AS feature_group_network
        FROM base_2024 b
        LEFT JOIN orig_2024 o ON b.dms_orig = o.dms_orig
        LEFT JOIN dest_2024 d ON b.dms_dest = d.dms_dest
        LEFT JOIN route_2024 r24 ON b.dms_orig = r24.dms_orig AND b.dms_dest = r24.dms_dest
        LEFT JOIN route_2023 r23 ON b.dms_orig = r23.dms_orig AND b.dms_dest = r23.dms_dest;
        """
    )
    feature_conn.commit()
    feature_conn.execute("DETACH DATABASE src")


def export_table_to_csv(conn: sqlite3.Connection, table: str, path: Path) -> None:
    if path.exists():
        path.unlink()
    query = f"SELECT {', '.join(EXPORT_COLS)} FROM {table}"
    for idx, chunk in enumerate(pd.read_sql_query(query, conn, chunksize=200_000), start=1):
        chunk.to_csv(path, mode="a", index=False, header=(idx == 1))
        print(f"[export {table} {idx}] rows={chunk.shape[0]:,}")


def summarize_table(conn: sqlite3.Connection, table: str) -> dict:
    result = {}
    result["row_count"] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    result["year_range"] = conn.execute(f"SELECT MIN(year), MAX(year) FROM {table}").fetchone()
    return result


def make_validation_artifacts(conn: sqlite3.Connection) -> dict:
    summary = {}
    nan_query = """
    SELECT
        SUM(CASE WHEN tons_lag1 IS NULL THEN 1 ELSE 0 END) AS tons_lag1_nan,
        SUM(CASE WHEN tons_lag2 IS NULL THEN 1 ELSE 0 END) AS tons_lag2_nan,
        SUM(CASE WHEN tons_growth_rate IS NULL THEN 1 ELSE 0 END) AS tons_growth_rate_nan,
        SUM(CASE WHEN value_per_ton_lag1 IS NULL THEN 1 ELSE 0 END) AS value_per_ton_lag1_nan,
        SUM(CASE WHEN tons_per_tmile_lag1 IS NULL THEN 1 ELSE 0 END) AS tons_per_tmile_lag1_nan
    FROM feature_all_rows;
    """
    summary["nan_counts"] = dict(
        zip(
            ["tons_lag1_nan", "tons_lag2_nan", "tons_growth_rate_nan", "value_per_ton_lag1_nan", "tons_per_tmile_lag1_nan"],
            conn.execute(nan_query).fetchone(),
        )
    )

    extreme_query = """
    SELECT
        SUM(CASE WHEN ABS(COALESCE(tons_growth_rate, 0)) > 10 THEN 1 ELSE 0 END) AS extreme_growth_rows,
        SUM(CASE WHEN value_per_ton_lag1 < 0 THEN 1 ELSE 0 END) AS negative_value_per_ton_rows,
        SUM(CASE WHEN tons_per_tmile_lag1 < 0 THEN 1 ELSE 0 END) AS negative_tons_per_tmile_rows
    FROM feature_model_ready;
    """
    summary["extreme_counts"] = dict(
        zip(
            ["extreme_growth_rows", "negative_value_per_ton_rows", "negative_tons_per_tmile_rows"],
            conn.execute(extreme_query).fetchone(),
        )
    )

    sample = pd.read_sql_query(
        """
        SELECT
            target_tons,
            tons_lag1,
            tons_lag2,
            tons_growth_rate,
            value_per_ton_lag1,
            tons_per_tmile_lag1,
            route_share_of_origin_lag1,
            route_share_of_dest_lag1
        FROM feature_model_ready
        WHERE rowid % 97 = 0
        LIMIT 120000;
        """,
        conn,
    )

    def dist_stats(series: pd.Series) -> dict[str, float | None]:
        clean = series.replace([math.inf, -math.inf], pd.NA).dropna()
        if clean.empty:
            return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
        return {
            "count": int(clean.shape[0]),
            "mean": float(clean.mean()),
            "median": float(clean.median()),
            "p95": float(clean.quantile(0.95)),
            "max": float(clean.max()),
        }

    metrics = [
        "target_tons",
        "tons_lag1",
        "tons_lag2",
        "tons_growth_rate",
        "value_per_ton_lag1",
        "tons_per_tmile_lag1",
        "route_share_of_origin_lag1",
        "route_share_of_dest_lag1",
    ]
    summary["distribution_stats"] = {metric: dist_stats(sample[metric]) for metric in metrics}

    plt.style.use("ggplot")
    plot_specs = [
        ("tons_lag1", "Distribution of tons_lag1", True),
        ("tons_growth_rate", "Distribution of tons_growth_rate", False),
        ("value_per_ton_lag1", "Distribution of value_per_ton_lag1", False),
        ("route_share_of_origin_lag1", "Distribution of route_share_of_origin_lag1", False),
    ]
    for col, title, log_x in plot_specs:
        clean = sample[col].replace([math.inf, -math.inf], pd.NA).dropna()
        if clean.empty:
            continue
        plt.figure(figsize=(8, 5))
        plot_values = clean.clip(lower=1e-9) if log_x else clean
        plt.hist(plot_values, bins=60, color="#33658a", alpha=0.9)
        if log_x:
            plt.xscale("log")
        plt.title(title)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(EDA_DIR / f"{col}.png", dpi=180)
        plt.close()

    return summary


def write_feature_docs(summary: dict) -> None:
    feature_lines = [
        "# Feature Description",
        "",
        "## Target",
        "- `target_tons`: target year truck tons.",
        "- `target_log_tons`: `log(1 + target_tons)` for skew-robust modeling.",
        "",
        "## Feature Groups",
        "- History: `tons_lag1`, `tons_lag2`, `tons_diff1`.",
        "- Trend: `tons_growth_rate`, `route_growth_rate`.",
        "- Cargo and distance: `sctg2`, `dist_band`, `trade_type`, `orig_state_fips`, `dest_state_fips`.",
        "- Value and efficiency: `current_value_lag1`, `tmiles_lag1`, `value_per_ton_lag1`, `tons_per_tmile_lag1`.",
        "- Time: `year`, `year_index`.",
        "- Network and hub context: `orig_total_outbound_tons_lag1`, `dest_total_inbound_tons_lag1`, `orig_unique_dest_count_lag1`, `dest_unique_orig_count_lag1`, `route_share_of_origin_lag1`, `route_share_of_dest_lag1`, `corridor_trade_type_count_lag1`, `corridor_sctg2_count_lag1`.",
        "",
        "## Modeling Notes",
        "- Lags were recomputed using the full row identity `(fr_orig, dms_orig, dms_dest, fr_dest, fr_inmode, dms_mode, fr_outmode, sctg2, trade_type, dist_band)` ordered by `year`.",
        "- This avoids the earlier leakage-like issue where lag values could drift across different routes inside the same year.",
        "- `feature_dataset_model_ready.csv` starts at year 2020 because `tons_lag2` is required.",
        "- `feature_dataset_forecast_2025.csv` is a scoring base with no target, built from 2024 and 2023 history.",
        "",
        "## Hub Simulation Relevance",
        "- `orig_total_outbound_tons_lag1` and `dest_total_inbound_tons_lag1` capture node scale.",
        "- `orig_unique_dest_count_lag1` and `dest_unique_orig_count_lag1` proxy node connectivity.",
        "- `route_share_of_origin_lag1` and `route_share_of_dest_lag1` proxy corridor dependence and can help flag fragile hub structures.",
        "- `corridor_trade_type_count_lag1` and `corridor_sctg2_count_lag1` describe route diversity, which is useful for resilience analysis.",
    ]
    FEATURE_DOC_MD.write_text("\n".join(feature_lines), encoding="utf-8")

    summary_lines = [
        "# Feature Summary",
        "",
        "## Outputs",
        f"- All rows CSV: `{FEATURE_ALL_CSV}`",
        f"- Model-ready CSV: `{FEATURE_MODEL_CSV}`",
        f"- Forecast base CSV: `{FEATURE_2025_CSV}`",
        f"- Feature DB: `{FEATURE_DB}`",
        f"- Validation plots: `{EDA_DIR}`",
        "",
        "## Row Counts",
        f"- All rows: {summary['all_rows']['row_count']:,}",
        f"- Model-ready rows: {summary['model_ready']['row_count']:,}",
        f"- Forecast 2025 rows: {summary['forecast_2025']['row_count']:,}",
        "",
        "## Validation",
        f"- NaN counts: {summary['validation']['nan_counts']}",
        f"- Extreme counts: {summary['validation']['extreme_counts']}",
    ]
    SUMMARY_MD.write_text("\n".join(summary_lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start = time.time()
    input_conn = connect(INPUT_DB)
    feature_conn = connect(FEATURE_DB)
    setup_feature_tables(feature_conn)
    build_feature_table(input_conn, feature_conn)
    build_model_ready_and_forecast(feature_conn)
    export_table_to_csv(feature_conn, "feature_all_rows", FEATURE_ALL_CSV)
    export_table_to_csv(feature_conn, "feature_model_ready", FEATURE_MODEL_CSV)
    export_table_to_csv(feature_conn, "feature_forecast_2025", FEATURE_2025_CSV)

    summary = {
        "generated_at_epoch": int(time.time()),
        "elapsed_seconds": round(time.time() - start, 2),
        "all_rows": summarize_table(feature_conn, "feature_all_rows"),
        "model_ready": summarize_table(feature_conn, "feature_model_ready"),
        "forecast_2025": summarize_table(feature_conn, "feature_forecast_2025"),
    }
    summary["validation"] = make_validation_artifacts(feature_conn)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_feature_docs(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYDEPS = PROJECT_ROOT / ".pydeps"
if str(PYDEPS) not in sys.path:
    sys.path.insert(0, str(PYDEPS))

import pandas as pd


FEATURE_DIR = PROJECT_ROOT / "outputs" / "features_truck_only"
INPUT_PATH = FEATURE_DIR / "feature_dataset_model_ready.csv"
OUTPUT_PATH = FEATURE_DIR / "feature_dataset_model_ready_with_splits.csv"
SUMMARY_PATH = FEATURE_DIR / "model_split_summary.json"


def assign_split(year: int) -> str:
    if year <= 2021:
        return "train"
    if year == 2022:
        return "validation"
    if year == 2023:
        return "test_2023"
    if year == 2024:
        return "test_2024"
    return "unused"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    df["split"] = df["year"].map(assign_split)
    df["is_train"] = (df["split"] == "train").astype(int)
    df["is_validation"] = (df["split"] == "validation").astype(int)
    df["is_test"] = df["split"].isin(["test_2023", "test_2024"]).astype(int)
    df.to_csv(OUTPUT_PATH, index=False)

    split_counts = df.groupby(["split", "year"]).size().reset_index(name="row_count")
    summary = {
        "input_rows": int(df.shape[0]),
        "output_path": str(OUTPUT_PATH),
        "split_counts": split_counts.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

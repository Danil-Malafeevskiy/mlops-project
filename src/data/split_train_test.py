"""Load raw ARFF, stratified train/test split, save Parquet under data/processed."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.json"


def load_split_params() -> dict:
    with open(PARAMS_PATH, encoding="utf-8") as f:
        return json.load(f)["split"]


def main() -> None:
    p = load_split_params()
    test_size = float(p["test_size"])
    random_state = int(p["random_state"])

    raw_path = ROOT / "data" / "raw" / "dataset"
    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, _ = arff.loadarff(raw_path)
    df = pd.DataFrame(data)
    df["Class"] = df["Class"].astype(int)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Class"],
    )

    train_path = out_dir / "train.parquet"
    test_path = out_dir / "test.parquet"
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"train: {len(train_df)} rows -> {train_path.relative_to(ROOT)}")
    print(f"test:  {len(test_df)} rows -> {test_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

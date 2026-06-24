from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "order_id",
    "order_date",
    "customer_id",
    "product",
    "category",
    "quantity",
    "unit_price",
    "discount",
    "region",
}


def generate_sample_data(rows: int = 2000, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    products = [
        ("Organic Milk", "Grocery", 4.8),
        ("Cold Brew", "Grocery", 5.5),
        ("Protein Bar", "Grocery", 2.9),
        ("Shampoo", "Beauty", 8.9),
        ("Face Serum", "Beauty", 21.5),
        ("Laundry Pods", "Home", 15.4),
    ]
    regions = ["North", "South", "East", "West", "Central"]
    records = []
    for idx in range(rows):
        product, category, price = products[rng.integers(0, len(products))]
        records.append(
            {
                "order_id": f"O{idx + 1000}",
                "order_date": rng.choice(dates),
                "customer_id": f"C{10000 + rng.integers(0, 500)}",
                "product": product,
                "category": category,
                "quantity": int(rng.poisson(2.2) + 1),
                "unit_price": round(float(price * rng.normal(1, 0.08)), 2),
                "discount": round(float(np.clip(rng.normal(0.09, 0.05), 0, 0.30)), 2),
                "region": rng.choice(regions),
            }
        )
    return pd.DataFrame(records)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output.columns = (
        output.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return output


def load_dataset(path_str: str | None) -> pd.DataFrame:
    if path_str:
        path = Path(path_str)
        if path.exists():
            return pd.read_csv(path)
    return generate_sample_data()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = normalize_columns(df).drop_duplicates().copy()
    missing = REQUIRED_COLUMNS - set(cleaned.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    for column in ["quantity", "unit_price", "discount"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=["order_date", "customer_id", "product", "category", "quantity", "unit_price"])
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)]
    cleaned["discount"] = cleaned["discount"].fillna(0).clip(0, 0.9)
    cleaned["region"] = cleaned["region"].fillna("Unknown")
    cleaned["revenue"] = (cleaned["quantity"] * cleaned["unit_price"] * (1 - cleaned["discount"])).round(2)
    cleaned["month"] = cleaned["order_date"].dt.to_period("M").dt.to_timestamp()
    cleaned["week"] = cleaned["order_date"].dt.to_period("W").apply(lambda value: value.start_time)
    return cleaned.sort_values("order_date")


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse ETL preparation step")
    parser.add_argument("--input", default=None, help="Optional input CSV path")
    parser.add_argument("--output", default="prepared_retailpulse.csv", help="Output CSV path")
    args = parser.parse_args()

    cleaned = clean_dataset(load_dataset(args.input))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()

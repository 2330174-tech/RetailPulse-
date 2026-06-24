from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(rows: int = 1500, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    products = ["Organic Milk", "Cold Brew", "Protein Bar", "Shampoo", "Face Serum", "Laundry Pods"]
    categories = {
        "Organic Milk": "Grocery",
        "Cold Brew": "Grocery",
        "Protein Bar": "Grocery",
        "Shampoo": "Beauty",
        "Face Serum": "Beauty",
        "Laundry Pods": "Home",
    }
    rows_out = []
    for idx in range(rows):
        product = products[rng.integers(0, len(products))]
        rows_out.append(
            {
                "order_id": f"O{idx}",
                "order_date": rng.choice(dates),
                "customer_id": f"C{1000 + rng.integers(0, 250)}",
                "product": product,
                "category": categories[product],
                "quantity": int(rng.poisson(2) + 1),
                "unit_price": round(float(rng.normal(12, 4)), 2),
                "discount": round(float(np.clip(rng.normal(0.08, 0.04), 0, 0.3)), 2),
                "region": rng.choice(["North", "South", "East", "West"]),
            }
        )
    df = pd.DataFrame(rows_out)
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)
    return df


def load_dataset(path_str: str | None, seed: int) -> pd.DataFrame:
    if path_str:
        path = Path(path_str)
        if path.exists():
            return pd.read_csv(path)
    return generate_sample_data(seed=seed)


def psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_ratio = np.clip(expected_counts / max(expected_counts.sum(), 1), 1e-6, None)
    actual_ratio = np.clip(actual_counts / max(actual_counts.sum(), 1), 1e-6, None)
    return float(np.sum((actual_ratio - expected_ratio) * np.log(actual_ratio / expected_ratio)))


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    expected_ratio = expected.astype(str).value_counts(normalize=True)
    actual_ratio = actual.astype(str).value_counts(normalize=True)
    keys = sorted(set(expected_ratio.index) | set(actual_ratio.index))
    total = 0.0
    for key in keys:
        e = max(float(expected_ratio.get(key, 0.0)), 1e-6)
        a = max(float(actual_ratio.get(key, 0.0)), 1e-6)
        total += (a - e) * math.log(a / e)
    return float(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse drift detection")
    parser.add_argument("--baseline", default=None, help="Baseline dataset CSV")
    parser.add_argument("--current", default=None, help="Current dataset CSV")
    parser.add_argument("--output", default="drift_report.json", help="Output JSON file")
    parser.add_argument("--threshold", type=float, default=0.2, help="PSI threshold for drift alert")
    parser.add_argument("--fail-on-drift", action="store_true", help="Exit non-zero when drift is detected")
    args = parser.parse_args()

    baseline = load_dataset(args.baseline, seed=21)
    current = load_dataset(args.current, seed=42)

    numeric_columns = [column for column in ["quantity", "unit_price", "discount", "revenue"] if column in baseline.columns and column in current.columns]
    categorical_columns = [column for column in ["category", "region", "product"] if column in baseline.columns and column in current.columns]

    report = {"threshold": args.threshold, "numeric": {}, "categorical": {}, "drift_detected": False}

    for column in numeric_columns:
        score = psi_numeric(baseline[column], current[column])
        report["numeric"][column] = {"psi": round(score, 4), "drift": score >= args.threshold}

    for column in categorical_columns:
        score = psi_categorical(baseline[column], current[column])
        report["categorical"][column] = {"psi": round(score, 4), "drift": score >= args.threshold}

    report["drift_detected"] = any(item["drift"] for item in report["numeric"].values()) or any(
        item["drift"] for item in report["categorical"].values()
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.fail_on_drift and report["drift_detected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

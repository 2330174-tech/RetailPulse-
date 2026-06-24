from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


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


def generate_sample_data(rows: int = 2000, seed: int = 42) -> pd.DataFrame:
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
    customers = [f"C{10000 + i}" for i in range(400)]
    records = []
    for idx in range(rows):
        product, category, price = products[rng.integers(0, len(products))]
        quantity = int(rng.poisson(2.0) + 1)
        discount = float(np.clip(rng.normal(0.08, 0.05), 0, 0.30))
        unit_price = float(round(price * rng.normal(1, 0.08), 2))
        records.append(
            {
                "order_id": f"O{200000 + idx}",
                "order_date": rng.choice(dates),
                "customer_id": rng.choice(customers),
                "product": product,
                "category": category,
                "quantity": quantity,
                "unit_price": unit_price,
                "discount": round(discount, 2),
                "region": rng.choice(regions),
            }
        )
    return pd.DataFrame(records)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return normalized


def load_dataset(input_path: str | None) -> pd.DataFrame:
    if input_path:
        path = Path(input_path)
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
    cleaned["revenue"] = (cleaned["quantity"] * cleaned["unit_price"] * (1 - cleaned["discount"])).round(2)
    cleaned["week"] = cleaned["order_date"].dt.to_period("W").apply(lambda value: value.start_time)
    return cleaned.sort_values("order_date")


def train_forecast_model(df: pd.DataFrame) -> tuple[LinearRegression, float, pd.DataFrame]:
    weekly = df.groupby("week", as_index=False)["quantity"].sum().rename(columns={"quantity": "actual_units"})
    weekly = weekly.sort_values("week").tail(60)
    if len(weekly) < 8:
        raise ValueError("Not enough weekly history to train the forecasting job.")
    weekly["t"] = np.arange(len(weekly))
    split_index = max(4, int(len(weekly) * 0.8))
    train = weekly.iloc[:split_index]
    test = weekly.iloc[split_index:]
    model = LinearRegression()
    model.fit(train[["t"]], train["actual_units"])
    predictions = model.predict(test[["t"]])
    mae = float(mean_absolute_error(test["actual_units"], predictions))
    results = test.copy()
    results["predicted_units"] = predictions.round(2)
    return model, mae, results


def build_customer_clusters(df: pd.DataFrame, n_clusters: int = 4) -> tuple[KMeans, pd.DataFrame]:
    max_date = df["order_date"].max()
    customer = (
        df.groupby("customer_id")
        .agg(
            recency_days=("order_date", lambda dates: (max_date - dates.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
            avg_discount=("discount", "mean"),
            units=("quantity", "sum"),
        )
        .reset_index()
    )
    features = ["recency_days", "frequency", "monetary", "avg_discount", "units"]
    scaled = StandardScaler().fit_transform(customer[features])
    model = KMeans(n_clusters=n_clusters, random_state=7, n_init=10)
    customer["cluster"] = model.fit_predict(scaled)
    return model, customer


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailPulse MLflow training job")
    parser.add_argument("--input", default=None, help="Optional CSV dataset path")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")
    parser.add_argument("--experiment", default="RetailPulse", help="MLflow experiment name")
    parser.add_argument("--artifact-dir", default="mlops_artifacts", help="Directory for generated artifacts")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)

    df = clean_dataset(load_dataset(args.input))
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    forecast_model, forecast_mae, forecast_results = train_forecast_model(df)
    cluster_model, customer_clusters = build_customer_clusters(df)

    summary = {
        "rows_after_cleaning": int(len(df)),
        "orders": int(df["order_id"].nunique()),
        "customers": int(df["customer_id"].nunique()),
        "categories": int(df["category"].nunique()),
        "forecast_mae": round(forecast_mae, 4),
        "cluster_count": int(cluster_model.n_clusters),
    }
    summary_path = artifact_dir / "training_summary.json"
    forecast_path = artifact_dir / "forecast_predictions.csv"
    cluster_path = artifact_dir / "customer_clusters.csv"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    forecast_results.to_csv(forecast_path, index=False)
    customer_clusters.to_csv(cluster_path, index=False)

    with mlflow.start_run(run_name="retailpulse_baseline_run"):
        mlflow.log_param("input_path", args.input or "generated_sample")
        mlflow.log_param("cluster_count", int(cluster_model.n_clusters))
        mlflow.log_metric("rows_after_cleaning", len(df))
        mlflow.log_metric("orders", df["order_id"].nunique())
        mlflow.log_metric("customers", df["customer_id"].nunique())
        mlflow.log_metric("forecast_mae", forecast_mae)
        mlflow.sklearn.log_model(forecast_model, artifact_path="forecast_model")
        mlflow.sklearn.log_model(cluster_model, artifact_path="cluster_model")
        mlflow.log_artifact(str(summary_path))
        mlflow.log_artifact(str(forecast_path))
        mlflow.log_artifact(str(cluster_path))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

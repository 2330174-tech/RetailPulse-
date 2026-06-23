from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="RetailPulse Analytics",
    page_icon="RP",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


@dataclass
class CleanResult:
    raw_rows: int
    clean_rows: int
    duplicate_rows: int
    invalid_rows: int
    missing_values: int


def generate_sample_data(rows: int = 4200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    products = [
        ("Organic Milk", "Grocery", 4.8),
        ("Cold Brew", "Grocery", 5.5),
        ("Protein Bar", "Grocery", 2.9),
        ("Shampoo", "Beauty", 8.9),
        ("Face Serum", "Beauty", 21.5),
        ("Body Lotion", "Beauty", 11.2),
        ("Laundry Pods", "Home", 15.4),
        ("Scented Candle", "Home", 13.0),
        ("Storage Basket", "Home", 18.8),
        ("Running Socks", "Apparel", 9.5),
        ("Rain Jacket", "Apparel", 44.0),
        ("Yoga Mat", "Sports", 29.0),
    ]
    regions = ["North", "South", "East", "West", "Central"]
    customers = [f"C{10000 + i}" for i in range(720)]
    product_weights = np.array([0.12, 0.11, 0.12, 0.09, 0.08, 0.08, 0.1, 0.08, 0.06, 0.06, 0.04, 0.06])
    product_weights = product_weights / product_weights.sum()

    affinity = {
        "Organic Milk": ["Protein Bar", "Cold Brew"],
        "Cold Brew": ["Protein Bar", "Running Socks"],
        "Shampoo": ["Body Lotion", "Face Serum"],
        "Face Serum": ["Body Lotion", "Scented Candle"],
        "Laundry Pods": ["Storage Basket", "Scented Candle"],
        "Yoga Mat": ["Running Socks", "Protein Bar"],
        "Rain Jacket": ["Running Socks", "Storage Basket"],
    }
    product_lookup = {name: (name, category, price) for name, category, price in products}
    product_names = [name for name, _, _ in products]

    records = []
    order_number = 200000
    while len(records) < rows:
        order_date = pd.Timestamp(rng.choice(dates))
        customer_id = rng.choice(customers)
        region = rng.choice(regions, p=[0.22, 0.2, 0.18, 0.24, 0.16])
        basket_size = int(rng.choice([1, 2, 3, 4], p=[0.48, 0.33, 0.15, 0.04]))
        first_product = rng.choice(product_names, p=product_weights)
        chosen = [first_product]
        for _ in range(basket_size - 1):
            related = affinity.get(chosen[-1], product_names)
            pool = [item for item in related if item not in chosen] or [item for item in product_names if item not in chosen]
            chosen.append(rng.choice(pool))

        weekend_lift = 1.2 if order_date.dayofweek >= 5 else 1.0
        seasonal_lift = 1.15 if order_date.month in [11, 12] else 1.0
        for product_name in chosen:
            _, category, base_price = product_lookup[product_name]
            quantity = int(rng.poisson(2.2 * weekend_lift * seasonal_lift) + 1)
            discount = float(np.clip(rng.normal(0.09, 0.06), 0, 0.35).round(2))
            unit_price = float(round(base_price * rng.normal(1, 0.08), 2))
            records.append(
                {
                    "order_id": f"O{order_number}",
                    "order_date": order_date,
                    "customer_id": customer_id,
                    "product": product_name,
                    "category": category,
                    "quantity": quantity,
                    "unit_price": unit_price,
                    "discount": discount,
                    "region": region,
                }
            )
            if len(records) >= rows:
                break
        order_number += 1

    df = pd.DataFrame(records)
    df["revenue"] = (df["quantity"] * df["unit_price"] * (1 - df["discount"])).round(2)

    # Add a few realistic quality issues so the cleaning panel has work to do.
    dirty = df.sample(frac=0.012, random_state=seed).index
    df.loc[dirty[: len(dirty) // 2], "quantity"] = -1
    df.loc[dirty[len(dirty) // 2 :], "region"] = None
    duplicate_rows = df.sample(18, random_state=seed + 1)
    return pd.concat([df, duplicate_rows], ignore_index=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return cleaned


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, CleanResult]:
    raw_rows = len(df)
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    cleaned = normalize_columns(df).drop_duplicates()

    cleaned["order_date"] = pd.to_datetime(cleaned["order_date"], errors="coerce")
    for column in ["quantity", "unit_price", "discount"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before_rules = len(cleaned)
    cleaned = cleaned.dropna(subset=["order_date", "customer_id", "product", "category", "quantity", "unit_price"])
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)]
    cleaned["discount"] = cleaned["discount"].fillna(0).clip(0, 0.9)
    cleaned["region"] = cleaned["region"].fillna("Unknown")
    cleaned["revenue"] = (cleaned["quantity"] * cleaned["unit_price"] * (1 - cleaned["discount"])).round(2)
    cleaned["month"] = cleaned["order_date"].dt.to_period("M").dt.to_timestamp()
    cleaned["week"] = cleaned["order_date"].dt.to_period("W").apply(lambda value: value.start_time)
    cleaned["basket_key"] = cleaned["order_id"].astype(str)

    invalid_rows = before_rules - len(cleaned)
    result = CleanResult(raw_rows, len(cleaned), duplicate_rows, invalid_rows, missing_values)
    return cleaned.sort_values("order_date"), result


def check_schema(df: pd.DataFrame) -> list[str]:
    columns = set(normalize_columns(df).columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    return missing


def product_pair_relationships(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    pairs = []
    baskets = df.groupby("basket_key")["product"].apply(lambda values: sorted(set(values)))
    for products in baskets:
        if len(products) < 2:
            continue
        for left_index, left in enumerate(products):
            for right in products[left_index + 1 :]:
                pairs.append((left, right))
    if not pairs:
        return pd.DataFrame(columns=["product_a", "product_b", "pair_count"])
    return (
        pd.DataFrame(pairs, columns=["product_a", "product_b"])
        .value_counts(["product_a", "product_b"])
        .reset_index(name="pair_count")
        .head(top_n)
    )


def forecast_weekly_demand(df: pd.DataFrame, category: str, horizon: int = 8) -> pd.DataFrame:
    source = df if category == "All" else df[df["category"] == category]
    weekly = source.groupby("week", as_index=False)["quantity"].sum().rename(columns={"quantity": "actual_units"})
    weekly = weekly.sort_values("week").tail(52)
    if len(weekly) < 4:
        return pd.DataFrame(columns=["week", "actual_units", "forecast_units", "type"])

    x = np.arange(len(weekly))
    y = weekly["actual_units"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    trailing = pd.Series(y).rolling(4, min_periods=1).mean().to_numpy()
    fitted = 0.45 * (slope * x + intercept) + 0.55 * trailing

    future_x = np.arange(len(weekly), len(weekly) + horizon)
    seasonal_index = np.resize(y[-8:] / max(np.mean(y[-8:]), 1), horizon)
    future = np.maximum(0, (slope * future_x + intercept) * seasonal_index)
    future_weeks = pd.date_range(weekly["week"].max() + pd.Timedelta(days=7), periods=horizon, freq="W-MON")

    history = weekly.assign(forecast_units=fitted.round(0), type="Historical")
    future_df = pd.DataFrame(
        {
            "week": future_weeks,
            "actual_units": np.nan,
            "forecast_units": future.round(0),
            "type": "Forecast",
        }
    )
    return pd.concat([history, future_df], ignore_index=True)


def build_customer_clusters(df: pd.DataFrame, n_clusters: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    customer = customer[customer["frequency"] > 0].copy()
    features = ["recency_days", "frequency", "monetary", "avg_discount", "units"]
    scaled = StandardScaler().fit_transform(customer[features])
    clusters = KMeans(n_clusters=n_clusters, random_state=7, n_init=10).fit_predict(scaled)
    customer["cluster"] = clusters

    profile = (
        customer.groupby("cluster")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_spend=("monetary", "mean"),
            avg_discount=("avg_discount", "mean"),
            avg_units=("units", "mean"),
        )
        .round(2)
        .reset_index()
    )
    spend_cutoff = profile["avg_spend"].median()
    frequency_cutoff = profile["avg_frequency"].median()
    discount_cutoff = profile["avg_discount"].median()
    recency_cutoff = profile["avg_recency"].median()
    profile["segment"] = profile.apply(
        lambda row: label_cluster(row, spend_cutoff, frequency_cutoff, discount_cutoff, recency_cutoff),
        axis=1,
    )
    customer = customer.merge(profile[["cluster", "segment"]], on="cluster", how="left")
    return customer, profile


def label_cluster(
    row: pd.Series,
    spend_cutoff: float,
    frequency_cutoff: float,
    discount_cutoff: float,
    recency_cutoff: float,
) -> str:
    if row["avg_spend"] >= spend_cutoff and row["avg_frequency"] >= frequency_cutoff:
        return "High value loyalists"
    if row["avg_discount"] >= discount_cutoff:
        return "Promotion responders"
    if row["avg_recency"] >= recency_cutoff:
        return "At-risk customers"
    return "Emerging regulars"


def make_insights(df: pd.DataFrame, clusters: pd.DataFrame, forecast: pd.DataFrame, pairs: pd.DataFrame) -> list[str]:
    revenue_by_category = df.groupby("category")["revenue"].sum().sort_values(ascending=False)
    top_category = revenue_by_category.index[0]
    top_category_share = revenue_by_category.iloc[0] / revenue_by_category.sum()

    region_growth = (
        df.groupby(["month", "region"])["revenue"].sum().reset_index().sort_values("month")
    )
    latest_month = region_growth["month"].max()
    prior_month = latest_month - pd.DateOffset(months=1)
    latest = region_growth[region_growth["month"] == latest_month].set_index("region")["revenue"]
    prior = region_growth[region_growth["month"] == prior_month].set_index("region")["revenue"]
    growth = ((latest - prior) / prior.replace(0, np.nan)).dropna().sort_values(ascending=False)
    top_region = growth.index[0] if not growth.empty else df["region"].mode().iloc[0]

    future_forecast = forecast[forecast["type"] == "Forecast"]["forecast_units"]
    demand_change = 0
    if len(future_forecast) >= 2:
        demand_change = (future_forecast.iloc[-1] - future_forecast.iloc[0]) / max(future_forecast.iloc[0], 1)

    biggest_segment = clusters.sort_values("customers", ascending=False).iloc[0]
    best_pair = None if pairs.empty else pairs.iloc[0]

    insights = [
        f"{top_category} leads revenue with {top_category_share:.0%} of cleaned sales, making it the first category to protect in assortment and replenishment plans.",
        f"{top_region} shows the strongest recent regional revenue momentum, so local promotions and inventory transfers should prioritize that market.",
        f"The largest customer cluster is {biggest_segment['segment']} with {int(biggest_segment['customers'])} customers, useful for tailored retention and lifecycle messaging.",
    ]
    if best_pair is not None:
        insights.append(
            f"{best_pair['product_a']} and {best_pair['product_b']} appear together most often, supporting bundled placement, cross-sell prompts, or shared promotion testing."
        )
    if demand_change > 0.08:
        insights.append("The forecast points to rising unit demand over the next eight weeks, so safety stock should increase for the selected category.")
    elif demand_change < -0.08:
        insights.append("The forecast points to softening demand, so markdowns and purchase orders should be reviewed before new inventory commitments.")
    else:
        insights.append("The forecast is relatively stable, which favors margin protection over aggressive discounting.")
    return insights


def download_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def sidebar_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    st.sidebar.title("RetailPulse")
    st.sidebar.caption("AI-powered customer analytics and demand forecasting")
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    category = st.sidebar.selectbox("Forecast category", categories)
    cluster_count = st.sidebar.slider("Customer clusters", 3, 6, 4)
    regions = st.sidebar.multiselect("Regions", sorted(df["region"].unique()), default=sorted(df["region"].unique()))
    filtered = df[df["region"].isin(regions)] if regions else df
    return filtered, category, cluster_count


def main() -> None:
    st.title("RetailPulse")
    st.caption("Clean retail data, reveal product relationships, forecast demand, cluster customers, and publish insights.")

    uploaded = st.sidebar.file_uploader("Upload retail CSV", type=["csv"])
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        raw = generate_sample_data()

    missing = check_schema(raw)
    if missing:
        st.error("The dataset is missing required columns: " + ", ".join(missing))
        st.stop()

    df, clean_result = clean_dataset(raw)
    filtered, category, cluster_count = sidebar_filters(df)
    customers, cluster_profile = build_customer_clusters(filtered, cluster_count)
    forecast = forecast_weekly_demand(filtered, category)
    pairs = product_pair_relationships(filtered)
    insights = make_insights(filtered, cluster_profile, forecast, pairs)

    kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
    kpi_a.metric("Clean revenue", f"${filtered['revenue'].sum():,.0f}")
    kpi_b.metric("Orders", f"{filtered['order_id'].nunique():,}")
    kpi_c.metric("Customers", f"{filtered['customer_id'].nunique():,}")
    kpi_d.metric("Forecast horizon", "8 weeks")

    tabs = st.tabs(["Dataset Cleaning", "Product Relationships", "Forecasting", "Customer Clusters", "Insights"])

    with tabs[0]:
        st.subheader("Dataset cleaning")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Raw rows", f"{clean_result.raw_rows:,}")
        c2.metric("Clean rows", f"{clean_result.clean_rows:,}")
        c3.metric("Duplicates removed", f"{clean_result.duplicate_rows:,}")
        c4.metric("Invalid rows removed", f"{clean_result.invalid_rows:,}")
        c5.metric("Missing values found", f"{clean_result.missing_values:,}")
        st.dataframe(filtered.head(250), use_container_width=True)
        st.download_button("Download cleaned CSV", download_csv(filtered), "retailpulse_cleaned.csv", "text/csv")

    with tabs[1]:
        st.subheader("Product relationship analysis")
        left, right = st.columns([1.15, 0.85])
        revenue_product = filtered.groupby(["product", "category"], as_index=False)["revenue"].sum().sort_values("revenue", ascending=False)
        left.plotly_chart(
            px.bar(revenue_product.head(15), x="revenue", y="product", color="category", orientation="h", title="Top products by revenue"),
            use_container_width=True,
        )
        if pairs.empty:
            right.info("Not enough multi-product baskets were found for pair analysis.")
        else:
            pair_fig = px.bar(
                pairs.sort_values("pair_count"),
                x="pair_count",
                y=pairs.apply(lambda row: f"{row['product_a']} + {row['product_b']}", axis=1),
                orientation="h",
                title="Frequently bought together",
            )
            right.plotly_chart(pair_fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Time-series demand forecasting")
        if forecast.empty:
            st.warning("Not enough weekly history to forecast the selected category.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=forecast["week"],
                    y=forecast["actual_units"],
                    mode="lines+markers",
                    name="Actual units",
                    line=dict(color="#5367b2", width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast["week"],
                    y=forecast["forecast_units"],
                    mode="lines+markers",
                    name="Forecast units",
                    line=dict(color="#0f8b8d", width=3, dash="dash"),
                )
            )
            fig.update_layout(title=f"Weekly demand forecast: {category}", xaxis_title="Week", yaxis_title="Units")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(forecast.tail(12), use_container_width=True)

    with tabs[3]:
        st.subheader("Customer clustering")
        left, right = st.columns([0.9, 1.1])
        scatter = px.scatter(
            customers,
            x="frequency",
            y="monetary",
            color="segment",
            size="units",
            hover_data=["customer_id", "recency_days", "avg_discount"],
            title="Behavioral customer segments",
        )
        left.plotly_chart(scatter, use_container_width=True)
        right.dataframe(cluster_profile, use_container_width=True)
        st.download_button("Download customer clusters", download_csv(customers), "retailpulse_customer_clusters.csv", "text/csv")

    with tabs[4]:
        st.subheader("Executive insights")
        for insight in insights:
            st.markdown(f"- {insight}")
        st.divider()
        st.markdown(
            "Recommended next step: connect live POS, inventory, promotion, and loyalty feeds, then schedule weekly model refreshes."
        )


if __name__ == "__main__":
    main()

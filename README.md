# RetailPulse Streamlit Analytics App

RetailPulse is a deployable analytics app for customer behavior and demand planning.

## Features

- Cleans retail transaction data by normalizing columns, removing duplicates, validating dates and numeric values, imputing safe defaults, and recalculating revenue.
- Visualizes product performance and frequently bought-together product relationships.
- Generates written business insights from category revenue, regional momentum, product pairings, customer clusters, and demand forecasts.
- Builds an eight-week time-series demand forecast using trend and recent seasonal behavior.
- Builds customer clusters with K-Means using recency, frequency, monetary value, discount sensitivity, and units purchased.
- Runs as a Streamlit website and supports CSV upload.

## Expected CSV Columns

```text
order_id, order_date, customer_id, product, category, quantity, unit_price, discount, region
```

If no CSV is uploaded, the app generates a realistic sample retail dataset automatically.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Deploy the `outputs/retailpulse_streamlit` folder to Streamlit Community Cloud, Render, or any host that supports Python web apps.

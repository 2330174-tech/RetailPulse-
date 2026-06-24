from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "retailpulse",
    "depends_on_past": False,
    "retries": 1,
}


with DAG(
    dag_id="retailpulse_daily_mlops",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["retailpulse", "etl", "mlops"],
) as dag:
    repo_root = "${RETAILPULSE_REPO_ROOT:-/opt/airflow}"

    extract_transform = BashOperator(
        task_id="extract_transform",
        bash_command=(
            f"python {repo_root}/mlops/etl_prepare_dataset.py "
            "--input ${RETAILPULSE_CURRENT_DATA:-} "
            "--output /tmp/retailpulse_prepared.csv"
        ),
    )

    train_and_log = BashOperator(
        task_id="train_and_log",
        bash_command=(
            f"python {repo_root}/mlops/mlflow_training_job.py "
            "--input /tmp/retailpulse_prepared.csv "
            "--tracking-uri ${MLFLOW_TRACKING_URI:-file:/tmp/mlruns}"
        ),
    )

    drift_check = BashOperator(
        task_id="drift_check",
        bash_command=(
            f"python {repo_root}/mlops/drift_detection.py "
            "--baseline ${RETAILPULSE_BASELINE_DATA:-} "
            "--current /tmp/retailpulse_prepared.csv "
            "--output /tmp/retailpulse_drift_report.json "
            "--fail-on-drift"
        ),
    )

    extract_transform >> train_and_log >> drift_check

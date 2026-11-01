
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="cs611_historical_monitoring",
    description="Date-specific retrospective loan monitoring",
    start_date=datetime(2024, 3, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["CS611", "MLE", "Monitoring"],
) as dag:

    historical_monitoring = BashOperator(
        task_id="run_historical_monitoring",
        bash_command=(
            'cd /app && python backfill_monitoring.py '
            '--month "$MONITOR_MONTH" '
            '--as-of "$MONITOR_AS_OF"'
        ),
        env={
            "MONITOR_MONTH": (
                "{{ dag_run.conf.get('month', '2024-03') "
                "if dag_run and dag_run.conf else '2024-03' }}"
            ),
            "MONITOR_AS_OF": (
                "{{ dag_run.conf.get('as_of', '2025-12-01') "
                "if dag_run and dag_run.conf else '2025-12-01' }}"
            ),
        },
        append_env=True,
    )

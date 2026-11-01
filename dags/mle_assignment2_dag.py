
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="cs611_loan_default_ml_pipeline",
    description="End-to-end loan default ML pipeline",
    start_date=datetime(2024, 3, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["CS611", "MLE", "Assignment2"],
) as dag:

    build_datamart = BashOperator(
        task_id="build_datamart",
        bash_command="cd /app && python main.py",
    )

    train_models = BashOperator(
        task_id="train_models",
        bash_command="cd /app && python train_models.py",
    )

    generate_predictions = BashOperator(
        task_id="generate_predictions",
        bash_command="cd /app && python predict_models.py",
    )

    monitor_models = BashOperator(
        task_id="monitor_models",
        bash_command="cd /app && python monitor_models.py",
    )


    create_charts = BashOperator(
        task_id="create_monitoring_charts",
        bash_command="cd /app && python create_monitoring_charts.py",
    )
  
   
    (
        build_datamart
        >> train_models
        >> generate_predictions
        >> monitor_models
        >> create_charts
    )
  
  
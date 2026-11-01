
# CS611 Machine Learning Engineering — Assignment 2

## Project Overview
An end-to-end loan default prediction pipeline using PySpark, scikit-learn, Docker and Apache Airflow.

The project extends Assignment 1's Bronze–Silver–Gold pipeline with feature engineering, model training, predictions and monitoring.

## Pipeline Architecture
1. `main.py` — Build Bronze, Silver and Gold layers
2. `train_models.py` — Train, compare and save models
3. `predict_models.py` — Generate predictions
4. `monitor_models.py` — Calculate performance and stability
5. `create_monitoring_charts.py` — Generate charts

## Running the Project
Requires Docker Desktop and Docker Compose.

```bash
docker compose up --build -d airflow
```

Open `http://localhost:8080`.

Retrieve the local Airflow admin password:

```bash
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```

Trigger `cs611_loan_default_ml_pipeline` in Airflow.

## Data and Features
Four CSV datasets supply customer attributes, financial information, clickstream activity and loan history.

The Gold feature store contains 12,500 unique loans.

Engineered features include financial ratios, repayment capacity, credit history, age groups and three-month historical clickstream averages.

Clickstream records on or after the application date are excluded.

## Model Training and Evaluation
Compared Logistic Regression and Random Forest using validation ROC-AUC.

Preprocessing includes median imputation, categorical encoding and vector assembly.

The selected model and preprocessing steps are saved in `model_bank/best_model`.

**Held-out test result:** 526 loans; ROC-AUC = 0.7277.

## Monitoring
Monitoring covers March 2024 to January 2025.

Metrics include ROC-AUC, accuracy, actual and predicted default rates, average risk score and PSI.

Output charts are saved in `charts/`.

## Historical Monitoring
The separate `cs611_historical_monitoring` DAG accepts a month and an evaluation cutoff date.

Example:

```bash
docker compose exec airflow airflow dags trigger --conf '{"month":"2024-03","as_of":"2025-12-01"}' cs611_historical_monitoring
```

Performance is calculated only for outcomes available by the requested cutoff.

## Limitations and Governance
- Historical monitoring is retrospective, not full point-in-time model retraining.
- The project uses a final-loan-balance-based default label.
- A fixed classification threshold may under-identify defaults.
- Score-distribution shifts require investigation.
- Age and occupation were excluded from the initial model training features; proxy fairness risks remain.
- Airflow standalone, SQLite and SequentialExecutor are used for local demonstration rather than production.

## Deliverables
- Python pipeline scripts and Airflow DAGs
- Docker configuration and dependencies
- Raw datasets and generated Gold outputs
- Saved trained models
- Monitoring charts and reports
- Presentation PDF

The main five-task DAG has been tested successfully end-to-end.
  
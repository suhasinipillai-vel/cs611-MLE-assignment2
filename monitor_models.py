
from pathlib import Path
from collections import defaultdict
from math import log
from datetime import date

from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("MLE_Assignment2_Monitoring")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    str(BASE_DIR / "datamart/gold/historical_predictions")
)

records = df.select(
    "prediction_month",
    "actual_default",
    "predicted_default",
    "default_probability",
    "label_snapshot_date"
).collect()

groups = defaultdict(list)

for row in records:
    groups[row["prediction_month"]].append(row)

def score_bins(values):
    counts = [0] * 10
    for value in values:
        bucket = min(9, max(0, int(float(value) * 10)))
        counts[bucket] += 1
    return counts

def calculate_psi(reference, current):
    a = score_bins(reference)
    b = score_bins(current)

    total_a = sum(a) + 5
    total_b = sum(b) + 5

    result = 0.0

    for x, y in zip(a, b):
        expected = (x + 0.5) / total_a
        actual = (y + 0.5) / total_b
        result += (actual - expected) * log(actual / expected)

    return result

reference_month = "2024-03"
reference_scores = [
    float(r["default_probability"])
    for r in groups[reference_month]
]

results = []
as_of_date = date.today()

for month in sorted(groups):
    month_rows = groups[month]

    scores = [
        float(r["default_probability"]) for r in month_rows
    ]

    known_rows = [
        r for r in month_rows
        if r["label_snapshot_date"] is not None
        and r["label_snapshot_date"] <= as_of_date
        and r["actual_default"] is not None
    ]

    actual = [int(r["actual_default"]) for r in known_rows]
    predicted = [int(r["predicted_default"]) for r in known_rows]
    known_scores = [
        float(r["default_probability"]) for r in known_rows
    ]

    if actual:
        accuracy = sum(
            a == p for a, p in zip(actual, predicted)
        ) / len(actual)

        actual_rate = sum(actual) / len(actual)
    else:
        accuracy = None
        actual_rate = None

    if len(set(actual)) == 2:
        auc = float(roc_auc_score(actual, known_scores))
    else:
        auc = None

    results.append({
        "month": month,
        "total_loans": len(month_rows),
        "evaluated_loans": len(known_rows),
        "actual_default_rate": actual_rate,
        "predicted_default_rate": sum(
            int(r["predicted_default"]) for r in month_rows
        ) / len(month_rows),
        "avg_default_probability": sum(scores) / len(scores),
        "accuracy": accuracy,
        "roc_auc": auc,
        "psi": calculate_psi(reference_scores, scores),
        "monitor_as_of_date": str(as_of_date)
    })

monitoring_df = spark.createDataFrame(results)

output_path = BASE_DIR / "datamart/gold/model_monitoring"

monitoring_df.write.mode("overwrite").parquet(
    str(output_path)
)

print("\nMONTHLY MODEL MONITORING RESULTS")

monitoring_df.select(
    "month",
    "total_loans",
    "roc_auc",
    "accuracy",
    "avg_default_probability",
    "psi"
).orderBy("month").show(20, truncate=False)

print("\nModel monitoring saved successfully!")

spark.stop()


import argparse
import json
from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--month", required=True)
parser.add_argument("--as-of", required=True)
args = parser.parse_args()

# Validate the supplied dates
datetime.strptime(args.month, "%Y-%m")
as_of_date = datetime.strptime(args.as_of, "%Y-%m-%d").date()

spark = (
    SparkSession.builder
    .appName("Historical_Monitoring_Backfill")
    .master("local[2]")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

# Existing retrospective predictions
df = spark.read.parquet(
    str(BASE_DIR / "datamart/gold/historical_predictions")
)

rows = df.filter(
    df.prediction_month == args.month
).collect()

if not rows:
    raise ValueError(f"No predictions for {args.month}")

# Only outcomes available by the requested cutoff
known = [
    r for r in rows
    if r["label_snapshot_date"] is not None
    and r["label_snapshot_date"] <= as_of_date
    and r["actual_default"] is not None
]

actual = [int(r["actual_default"]) for r in known]
scores = [float(r["default_probability"]) for r in known]

auc = (
    float(roc_auc_score(actual, scores))
    if len(set(actual)) == 2 else None
)

report = {
    "prediction_month": args.month,
    "as_of_date": args.as_of,
    "analysis_type": "retrospective_monitoring",
    "total_predictions": len(rows),
    "known_outcomes": len(known),
    "average_risk_score": sum(
        float(r["default_probability"]) for r in rows
    ) / len(rows),
    "actual_default_rate": (
        sum(actual) / len(actual) if actual else None
    ),
    "roc_auc": auc
}

output_dir = BASE_DIR / "reports" / "backfill"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / (
    f"{args.month}_asof_{args.as_of}.json"
)

with open(output_file, "w") as f:
    json.dump(report, f, indent=2)

print("\nHISTORICAL MONITORING REPORT")
print(json.dumps(report, indent=2))
print("Backfill monitoring report saved!")

spark.stop()

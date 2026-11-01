
from pathlib import Path
from pyspark.sql import SparkSession
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(exist_ok=True)

spark = (
    SparkSession.builder
    .appName("MLE_Monitoring_Charts")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet(
    str(BASE_DIR / "datamart/gold/model_monitoring")
)

rows = df.orderBy("month").collect()

months = [r["month"] for r in rows]

def save_chart(filename, title, series, ylabel):
    plt.figure(figsize=(11, 5))

    for label, values in series.items():
        plt.plot(months, values, marker="o", label=label)

    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.25)

    if len(series) > 1:
        plt.legend()

    plt.figtext(
        0.5, 0.01,
        "Retrospective analysis, not live deployment results",
        ha="center", fontsize=8
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(CHART_DIR / filename, dpi=200)
    plt.close()

save_chart(
    "roc_auc_trend.png",
    "Monthly Model Performance",
    {"ROC-AUC": [r["roc_auc"] for r in rows]},
    "ROC-AUC"
)

save_chart(
    "psi_trend.png",
    "Monthly Prediction Stability",
    {"PSI": [r["psi"] for r in rows]},
    "Population Stability Index"
)

save_chart(
    "default_rates.png",
    "Actual vs Predicted Default Rates",
    {
        "Actual": [
            r["actual_default_rate"] for r in rows
        ],
        "Predicted": [
            r["predicted_default_rate"] for r in rows
        ]
    },
    "Default rate"
)

print("Three monitoring charts saved successfully!")

spark.stop()

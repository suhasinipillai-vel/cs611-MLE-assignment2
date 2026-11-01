
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.evaluation import BinaryClassificationEvaluator

BASE_DIR = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("MLE_Assignment2_Predictions")
    .master("local[2]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

features = spark.read.parquet(
    str(BASE_DIR / "datamart/gold/feature_store")
)

labels = spark.read.parquet(
    str(BASE_DIR / "datamart/gold/label_store")
)

dataset = features.join(
    labels.select("loan_id", "label_snapshot_date", "default"),
    on="loan_id",
    how="inner"
)

test_df = (
    dataset
    .filter(F.col("feature_snapshot_date") >= "2025-01-01")
    .withColumn("label", F.col("default").cast("double"))
)

model = PipelineModel.load(
    str(BASE_DIR / "model_bank/best_model")
)

predictions = model.transform(test_df)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)

gold_predictions = predictions.select(
    "loan_id",
    "Customer_ID",
    "feature_snapshot_date",
    "label_snapshot_date",
    F.col("label").cast("int").alias("actual_default"),
    F.col("prediction").cast("int").alias("predicted_default"),
    vector_to_array("probability")[1].alias("default_probability")
)

gold_predictions.write.mode("overwrite").parquet(
    str(BASE_DIR / "datamart/gold/model_predictions")
)

print("\nTEST RESULTS")
print("Test loans:", gold_predictions.count())
print("Test ROC-AUC:", round(auc, 4))

gold_predictions.show(10, truncate=False)


print("\nGenerating historical predictions...")

history_df = (
    dataset
    .filter(F.col("feature_snapshot_date") >= F.lit("2024-03-01"))
    .withColumn("label", F.col("default").cast("double"))
)

history_scored = model.transform(history_df)

historical_predictions = history_scored.select(
    "loan_id",
    "Customer_ID",
    "feature_snapshot_date",
    "label_snapshot_date",
    F.date_format(
        "feature_snapshot_date", "yyyy-MM"
    ).alias("prediction_month"),
    F.col("label").cast("int").alias("actual_default"),
    F.col("prediction").cast("int").alias("predicted_default"),
    vector_to_array("probability")[1].alias("default_probability"),
    F.lit("best_model_v1").alias("model_version")
)

historical_predictions.write.mode("overwrite").partitionBy(
    "prediction_month"
).parquet(
    str(BASE_DIR / "datamart/gold/historical_predictions")
)

print("\nHISTORICAL PREDICTIONS BY MONTH")
historical_predictions.groupBy(
    "prediction_month"
).count().orderBy("prediction_month").show(20, False)

print("Historical predictions saved successfully!")
  
spark.stop()

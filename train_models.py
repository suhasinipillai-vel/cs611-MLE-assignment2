
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


BASE_DIR = Path(__file__).resolve().parent

spark = (
    SparkSession.builder
    .appName("MLE_Assignment2_Training")
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

print("Total training candidates:", dataset.count())


dataset.groupBy(
    F.year("feature_snapshot_date").alias("application_year")
).agg(
    F.count("*").alias("loans"),
    F.min("label_snapshot_date").alias("earliest_outcome"),
    F.max("label_snapshot_date").alias("latest_outcome")
).orderBy("application_year").show(truncate=False)


train_df = dataset.filter(
    F.col("label_snapshot_date") < F.lit("2024-01-01")
)

validation_df = dataset.filter(
    (F.col("feature_snapshot_date") >= F.lit("2024-01-01")) &
    (F.col("feature_snapshot_date") < F.lit("2024-03-01")) &
    (F.col("label_snapshot_date") < F.lit("2025-01-01"))
)

test_df = dataset.filter(
    F.col("feature_snapshot_date") >= F.lit("2025-01-01")
)

print("\nTRAINING DATA")
print("Rows:", train_df.count())
train_df.groupBy("default").count().show()

print("\nVALIDATION DATA")
print("Rows:", validation_df.count())
validation_df.groupBy("default").count().show()

print("\nTEST DATA")
print("Rows:", test_df.count())
test_df.groupBy("default").count().show()



from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

numeric_cols = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Outstanding_Debt",
    "Total_EMI_per_month",
    "loan_amt",
    "tenure",
    "debt_to_income_ratio",
    "emi_to_income_ratio",
    "repayment_capacity",
    "credit_history_months",
    "loan_to_income_ratio",
    "click_months_seen"
] + [f"click_mean_fe_{i}" for i in range(1, 21)]

categorical_cols = ["Credit_Mix", "Payment_Behaviour"]

train_ready = train_df.withColumn(
    "label", F.col("default").cast("double")
)
validation_ready = validation_df.withColumn(
    "label", F.col("default").cast("double")
)

imputed_cols = [f"imputed_{c}" for c in numeric_cols]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=imputed_cols,
    strategy="median"
)

indexed_cols = [f"{c}_idx" for c in categorical_cols]

indexer = StringIndexer(
    inputCols=categorical_cols,
    outputCols=indexed_cols,
    handleInvalid="keep"
)

encoded_cols = [f"{c}_encoded" for c in categorical_cols]

encoder = OneHotEncoder(
    inputCols=indexed_cols,
    outputCols=encoded_cols,
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=imputed_cols + encoded_cols,
    outputCol="features"
)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=80,
    regParam=0.1
)

pipeline = Pipeline(
    stages=[imputer, indexer, encoder, assembler, lr]
)

print("\nTraining Logistic Regression...")
model = pipeline.fit(train_ready)

predictions = model.transform(validation_ready)

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)

print("Validation ROC-AUC:", round(auc, 4))

print("\nValidation prediction results:")
predictions.groupBy("label", "prediction").count().show()

MODEL_DIR = BASE_DIR / "model_bank"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model.write().overwrite().save(
    str(MODEL_DIR / "logistic_regression")
)

print("Logistic Regression model saved successfully!")


from pyspark.ml.classification import RandomForestClassifier

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    numTrees=40,
    maxDepth=5,
    seed=42
)

rf_pipeline = Pipeline(
    stages=[imputer, indexer, encoder, assembler, rf]
)

print("\nTraining Random Forest...")
rf_model = rf_pipeline.fit(train_ready)

rf_predictions = rf_model.transform(validation_ready)
rf_auc = evaluator.evaluate(rf_predictions)

print("\nMODEL COMPARISON")
print("Logistic Regression ROC-AUC:", round(auc, 4))
print("Random Forest ROC-AUC:", round(rf_auc, 4))

rf_model.write().overwrite().save(
    str(MODEL_DIR / "random_forest")
)

if rf_auc > auc:
    best_model = rf_model
    best_model_name = "Random Forest"
else:
    best_model = model
    best_model_name = "Logistic Regression"

best_model.write().overwrite().save(
    str(MODEL_DIR / "best_model")
)

print("\nBest model:", best_model_name)
print("Best model saved successfully!")
  
spark.stop()

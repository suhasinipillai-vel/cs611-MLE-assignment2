from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATAMART_DIR = BASE_DIR / "datamart"
BRONZE_DIR = DATAMART_DIR / "bronze"
SILVER_DIR = DATAMART_DIR / "silver"
GOLD_DIR = DATAMART_DIR / "gold"

BRONZE_DIR.mkdir(parents=True, exist_ok=True)
SILVER_DIR.mkdir(parents=True, exist_ok=True)
GOLD_DIR.mkdir(parents=True, exist_ok=True)

spark = (
    SparkSession.builder
    .appName("Assignment1DataPipeline")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("Spark started successfully!")

raw_files = {
    "feature_clickstream": "feature_clickstream.csv",
    "features_attributes": "features_attributes.csv",
    "features_financials": "features_financials.csv",
    "lms_loan_daily": "lms_loan_daily.csv"
}

print("\nBuilding Bronze layer")

for table_name, file_name in raw_files.items():

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", False)
        .csv(str(DATA_DIR / file_name))
    )

    df.write.mode("overwrite").parquet(
        str(BRONZE_DIR / table_name)
    )

    print(f"{table_name}: {df.count()} rows")

print("Bronze layer completed!")


def clean_numeric(column_name, data_type="double"):

    cleaned = F.trim(
        F.regexp_replace(F.col(column_name), "_", "")
    )

    return (
        F.when(
            cleaned.rlike(r"^-?\d+(\.\d+)?$"),
            cleaned.cast(data_type)
        )
        .otherwise(F.lit(None).cast(data_type))
    )


print("\nBuilding Silver layer")

attributes = spark.read.parquet(
    str(BRONZE_DIR / "features_attributes")
)

attributes_silver = (
    attributes
    .withColumn(
        "snapshot_date",
        F.to_date("snapshot_date", "yyyy-MM-dd")
    )
    .withColumn(
        "Age",
        clean_numeric("Age", "int")
    )
    .withColumn(
        "Age",
        F.when(
            (F.col("Age") >= 18) &
            (F.col("Age") <= 100),
            F.col("Age")
        )
    )
    .withColumn(
        "Occupation",
        F.when(
            F.col("Occupation") == "_______",
            None
        ).otherwise(F.col("Occupation"))
    )
    .withColumn(
        "SSN",
        F.when(
            F.col("SSN").rlike(r"^\d{3}-\d{2}-\d{4}$"),
            F.col("SSN")
        )
    )
)

attributes_silver.write.mode("overwrite").parquet(
    str(SILVER_DIR / "features_attributes")
)

print(
    f"features_attributes: "
    f"{attributes_silver.count()} rows"
)


financials = spark.read.parquet(
    str(BRONZE_DIR / "features_financials")
)

financial_double_columns = [
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Interest_Rate",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance"
]

financial_integer_columns = [
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment"
]

financials_silver = financials

for column_name in financial_double_columns:

    financials_silver = financials_silver.withColumn(
        column_name,
        clean_numeric(column_name, "double")
    )

for column_name in financial_integer_columns:

    financials_silver = financials_silver.withColumn(
        column_name,
        clean_numeric(column_name, "int")
    )

financials_silver = (
    financials_silver
    .withColumn(
        "snapshot_date",
        F.to_date("snapshot_date", "yyyy-MM-dd")
    )
    .withColumn(
        "Credit_Mix",
        F.when(
            F.col("Credit_Mix") == "_",
            None
        ).otherwise(F.col("Credit_Mix"))
    )
    .withColumn(
        "Payment_Behaviour",
        F.when(
            F.col("Payment_Behaviour") == "!@9#%8",
            None
        ).otherwise(F.col("Payment_Behaviour"))
    )
)

financials_silver.write.mode("overwrite").parquet(
    str(SILVER_DIR / "features_financials")
)

print(
    f"features_financials: "
    f"{financials_silver.count()} rows"
)


clickstream = spark.read.parquet(
    str(BRONZE_DIR / "feature_clickstream")
)

clickstream_silver = clickstream.withColumn(
    "snapshot_date",
    F.to_date("snapshot_date", "yyyy-MM-dd")
)

for i in range(1, 21):

    clickstream_silver = (
        clickstream_silver
        .withColumn(
            f"fe_{i}",
            clean_numeric(f"fe_{i}", "double")
        )
    )

clickstream_silver.write.mode("overwrite").parquet(
    str(SILVER_DIR / "feature_clickstream")
)

print(
    f"feature_clickstream: "
    f"{clickstream_silver.count()} rows"
)


loans = spark.read.parquet(
    str(BRONZE_DIR / "lms_loan_daily")
)

loans_silver = (
    loans
    .withColumn(
        "loan_start_date",
        F.to_date("loan_start_date", "yyyy-MM-dd")
    )
    .withColumn(
        "snapshot_date",
        F.to_date("snapshot_date", "yyyy-MM-dd")
    )
    .withColumn(
        "tenure",
        clean_numeric("tenure", "int")
    )
    .withColumn(
        "installment_num",
        clean_numeric("installment_num", "int")
    )
    .withColumn(
        "loan_amt",
        clean_numeric("loan_amt", "double")
    )
    .withColumn(
        "due_amt",
        clean_numeric("due_amt", "double")
    )
    .withColumn(
        "paid_amt",
        clean_numeric("paid_amt", "double")
    )
    .withColumn(
        "overdue_amt",
        clean_numeric("overdue_amt", "double")
    )
    .withColumn(
        "balance",
        clean_numeric("balance", "double")
    )
)

loans_silver.write.mode("overwrite").parquet(
    str(SILVER_DIR / "lms_loan_daily")
)

print(
    f"lms_loan_daily: "
    f"{loans_silver.count()} rows"
)

print("Silver layer completed!")


print("\nBuilding Gold layer")

loan_applications = (
    loans_silver
    .filter(
        F.col("installment_num") == 0
    )
    .select(
        "loan_id",
        "Customer_ID",
        "loan_start_date",
        "loan_amt",
        "tenure"
    )
)

clickstream_3m = (
    loan_applications.alias("l")
    .join(
        clickstream_silver.alias("c"),
        (
            (F.col("l.Customer_ID") == F.col("c.Customer_ID"))
            &
            (F.col("c.snapshot_date") < F.col("l.loan_start_date"))
            &
            (
                F.col("c.snapshot_date") >=
                F.add_months(F.col("l.loan_start_date"), -3)
            )
        ),
        "left"
    )
    .groupBy(F.col("l.loan_id").alias("loan_id"))
    .agg(
        F.count("c.snapshot_date").alias("click_months_seen"),
        *[
            F.avg(F.col(f"c.fe_{i}")).alias(f"click_mean_fe_{i}")
            for i in range(1, 21)
        ]
    )
)

feature_store = (
    loan_applications.alias("l")

    .join(
        attributes_silver.alias("a"),
        (
            (
                F.col("l.Customer_ID") ==
                F.col("a.Customer_ID")
            )
            &
            (
                F.col("l.loan_start_date") ==
                F.col("a.snapshot_date")
            )
        ),
        "left"
    )

    .join(
        financials_silver.alias("f"),
        (
            (
                F.col("l.Customer_ID") ==
                F.col("f.Customer_ID")
            )
            &
            (
                F.col("l.loan_start_date") ==
                F.col("f.snapshot_date")
            )
        ),
        "left"
    )
  
    .join(
        clickstream_3m.alias("c"),
        F.col("l.loan_id") == F.col("c.loan_id"),
        "left"
    )
  
    .select(
        F.col("l.loan_id").alias("loan_id"),

        F.col("c.click_months_seen").alias("click_months_seen"),
        *[
            F.col(f"c.click_mean_fe_{i}").alias(f"click_mean_fe_{i}")
            for i in range(1, 21)
        ],
        F.col("l.Customer_ID").alias("Customer_ID"),
        F.col("l.loan_start_date").alias(
            "feature_snapshot_date"
        ),

        F.col("a.Age").alias("Age"),


        F.col("a.Occupation").alias("Occupation"),
  

        F.col("f.Annual_Income").alias(
            "Annual_Income"
        ),


         F.col("f.Credit_History_Age").alias("Credit_History_Age"), 
         F.col("f.Credit_Mix").alias("Credit_Mix"),
         F.col("f.Payment_Behaviour").alias("Payment_Behaviour"),
  
  
        F.col("f.Monthly_Inhand_Salary").alias(
            "Monthly_Inhand_Salary"
        ),

        F.col("f.Num_Bank_Accounts").alias(
            "Num_Bank_Accounts"
        ),

        F.col("f.Num_Credit_Card").alias(
            "Num_Credit_Card"
        ),

        F.col("f.Interest_Rate").alias(
            "Interest_Rate"
        ),

        F.col("f.Num_of_Loan").alias(
            "Num_of_Loan"
        ),

        F.col("f.Delay_from_due_date").alias(
            "Delay_from_due_date"
        ),

        F.col("f.Num_of_Delayed_Payment").alias(
            "Num_of_Delayed_Payment"
        ),

        F.col("f.Num_Credit_Inquiries").alias(
            "Num_Credit_Inquiries"
        ),

        F.col("f.Outstanding_Debt").alias(
            "Outstanding_Debt"
        ),

        F.col("f.Credit_Utilization_Ratio").alias(
            "Credit_Utilization_Ratio"
        ),

        F.col("f.Total_EMI_per_month").alias(
            "Total_EMI_per_month"
        ),

        F.col("f.Monthly_Balance").alias(
            "Monthly_Balance"
        ),

        F.col("l.loan_amt").alias(
            "loan_amt"
        ),

        F.col("l.tenure").alias(
            "tenure"
        )
    )

    .withColumn(
        "debt_to_income_ratio",
        F.when(
            F.col("Annual_Income") > 0,
            F.col("Outstanding_Debt") /
            F.col("Annual_Income")
        )
    )

    .withColumn(
        "emi_to_income_ratio",
        F.when(
            F.col("Monthly_Inhand_Salary") > 0,
            F.col("Total_EMI_per_month") /
            F.col("Monthly_Inhand_Salary")
        )
    )
)

feature_store = feature_store.withColumn(
    "repayment_capacity",
    F.col("Monthly_Inhand_Salary") - F.col("Total_EMI_per_month")
)

feature_store = feature_store.withColumn(
    "credit_history_months",
    F.when(
        F.col("Credit_History_Age").rlike(
            r"(?i)^\s*\d+\s+Years?\s+and\s+\d+\s+Months?\s*$"
        ),
        F.regexp_extract(
            "Credit_History_Age", r"(?i)(\d+)\s+Years?", 1
        ).cast("int") * 12
        +
        F.regexp_extract(
            "Credit_History_Age", r"(?i)(\d+)\s+Months?", 1
        ).cast("int")
    )
)
feature_store = feature_store.withColumn(
    "age_group",
    F.when(F.col("Age").between(18, 24), "18-24")
     .when(F.col("Age").between(25, 39), "25-39")
     .when(F.col("Age").between(40, 59), "40-59")
     .when(F.col("Age").between(60, 100), "60+")
     .otherwise(None)
)


feature_store = feature_store.withColumn(
    "loan_to_income_ratio",
    F.when(
        F.col("Annual_Income") > 0,
        F.col("loan_amt") / F.col("Annual_Income")
    )
)

feature_store.write.mode("overwrite").parquet(
    str(GOLD_DIR / "feature_store")
)

print(
    f"feature_store: "
    f"{feature_store.count()} rows"
)


max_installments = (
    loans_silver
    .groupBy("loan_id")
    .agg(
        F.max("installment_num").alias(
            "max_installment_num"
        )
    )
)

final_loan_status = (
    loans_silver.alias("l")

    .join(
        max_installments.alias("m"),
        (
            (
                F.col("l.loan_id") ==
                F.col("m.loan_id")
            )
            &
            (
                F.col("l.installment_num") ==
                F.col("m.max_installment_num")
            )
        ),
        "inner"
    )
)

label_store = (
    final_loan_status

    .withColumn(
        "default",
        F.when(
            F.col("l.balance") > 0,
            1
        ).otherwise(0)
    )

    .select(
        F.col("l.loan_id").alias(
            "loan_id"
        ),

        F.col("l.Customer_ID").alias(
            "Customer_ID"
        ),

        F.col("l.snapshot_date").alias(
            "label_snapshot_date"
        ),

        "default"
    )
)

label_store.write.mode("overwrite").parquet(
    str(GOLD_DIR / "label_store")
)

print(
    f"label_store: "
    f"{label_store.count()} rows"
)

print("\nLabel distribution")

label_store.groupBy(
    "default"
).count().orderBy(
    "default"
).show()

print("Gold layer completed!")

spark.stop()

print("\nPipeline completed successfully!")
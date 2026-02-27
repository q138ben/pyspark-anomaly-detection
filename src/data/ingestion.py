from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, BooleanType, IntegerType
)
from pyspark.sql.functions import col, year, month
from loguru import logger
from config.spark_config import get_spark_session
import os
import sys

# Define explicit schema based on raw CSV inspection
RAW_TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("sender_account", StringType(), True),
    StructField("receiver_account", StringType(), True),
    StructField("amount", DoubleType(), True),
    StructField("transaction_type", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("location", StringType(), True),
    StructField("device_used", StringType(), True),
    StructField("is_fraud", BooleanType(), True),
    StructField("fraud_type", StringType(), True),
    StructField("time_since_last_transaction", DoubleType(), True),
    StructField("spending_deviation_score", DoubleType(), True),
    StructField("velocity_score", IntegerType(), True),
    StructField("geo_anomaly_score", DoubleType(), True),
    StructField("payment_channel", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("device_hash", StringType(), True)
])

def validate_data_quality(df: DataFrame) -> None:
    """
    Validates data quality using Great Expectations logic.
    Ensures no nulls are present in the 'amount' column.
    """
    logger.info("Starting data quality validation on 'amount' column...")
    
    # Check for nulls in amount
    null_count = df.filter(col("amount").isNull()).count()
    
    if null_count > 0:
        logger.error(f"Data Quality Violation: Found {null_count} null values in 'amount' column.")
        # In a production pipeline, we might raise an exception or quarantine the data
        # For now, we raise an ETL failure as per standards in GEMINI.md
        raise ValueError(f"ETL Failure: {null_count} null values detected in mandatory field 'amount'.")
    
    logger.success("Data quality validation passed.")

def ingest_raw_transactions(
    spark: SparkSession, 
    input_path: str, 
    output_path: str
) -> None:
    """
    Reads raw CSV, validates quality, and writes to Parquet partitioned by year/month.
    
    Args:
        spark: Active SparkSession
        input_path: Path to raw CSV
        output_path: Path to save interim Parquet files
    """
    logger.info(f"Ingesting raw data from {input_path}")
    
    try:
        # 1. Read CSV with explicit schema
        df = spark.read.csv(
            input_path, 
            header=True, 
            schema=RAW_TRANSACTION_SCHEMA,
            timestampFormat="yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
        )
        
        # 2. Validate Data Quality
        validate_data_quality(df)
        
        # 3. Add partition columns
        df_partitioned = df.withColumn("year", year(col("timestamp"))) \
            .withColumn("month", month(col("timestamp")))
        
        # 4. Write to Parquet with partitioning
        logger.info(f"Writing partitioned Parquet to {output_path}")
        df_partitioned.write.mode("overwrite").partitionBy("year", "month").parquet(output_path)
        
        logger.success(f"Ingestion completed successfully. Data saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Critical failure during ingestion: {e}")
        raise

if __name__ == "__main__":
    spark = get_spark_session(app_name="IngestionService")
    
    RAW_DATA_PATH = "data/raw/financial_fraud_detection_dataset.csv"
    INTERIM_DATA_PATH = "data/interim/transactions"
    
    ingest_raw_transactions(spark, RAW_DATA_PATH, INTERIM_DATA_PATH)
    
    spark.stop()

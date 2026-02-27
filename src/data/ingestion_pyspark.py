from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, to_date
from loguru import logger
from config.spark_config import get_spark_session
from src.data.schemas import TRANSACTION_SCHEMA
import os
import sys

def ingest_to_bronze(
    spark: SparkSession, 
    input_path: str, 
    bronze_path: str, 
    quarantine_path: str
) -> None:
    """
    Reads raw CSV data, handles corrupt records, and writes to a Delta Lake table.
    
    Args:
        spark: Active SparkSession.
        input_path: Path to the raw CSV data.
        bronze_path: Output path for clean records (Delta format).
        quarantine_path: Output path for corrupted records.
    """
    logger.info(f"Ingesting raw data from {input_path}")
    
    try:
        # 1. Load data using explicit schema and columnNameOfCorruptRecord
        raw_df = (
            spark.read
            .option("header", "true")
            .option("mode", "PERMISSIVE") # Continue processing even if corrupt rows are found
            .option("columnNameOfCorruptRecord", "_corrupt_record")
            .schema(TRANSACTION_SCHEMA)
            .csv(input_path)
        )
        
        # 2. Separate clean records from corrupted records
        corrupt_df = raw_df.filter(col("_corrupt_record").isNotNull())
        clean_df = raw_df.filter(col("_corrupt_record").isNull()).drop("_corrupt_record")
        
        # 3. Handle quarantined records
        if corrupt_df.count() > 0:
            logger.warning(f"Detected {corrupt_df.count()} corrupt records. Writing to quarantine: {quarantine_path}")
            corrupt_df.write.mode("append").parquet(quarantine_path)
        
        # 4. Partition clean data by date and write to Delta
        if clean_df.count() > 0:
            logger.info(f"Writing {clean_df.count()} clean records to Bronze layer (Delta): {bronze_path}")
            
            # Use to_date for partition column
            clean_df_with_date = clean_df.withColumn("ingestion_date", to_date(col("timestamp")))
            
            (
                clean_df_with_date.write
                .format("delta")
                .mode("overwrite")
                .partitionBy("ingestion_date")
                .save(bronze_path)
            )
            logger.success("Bronze ingestion successful.")
        else:
            logger.warning("No clean records found to ingest.")
            
    except Exception as e:
        logger.exception(f"Critical error during bronze ingestion: {e}")
        sys.exit(1)

if __name__ == "__main__":
    spark = get_spark_session(app_name="BronzeIngestion")
    
    # Define paths
    RAW_PATH = "data/raw/financial_fraud_detection_dataset.csv"
    BRONZE_PATH = "data/bronze/transactions"
    QUARANTINE_PATH = "data/bronze/quarantine"
    
    # Ensure directories exist
    os.makedirs("data/bronze", exist_ok=True)
    
    ingest_to_bronze(spark, RAW_PATH, BRONZE_PATH, QUARANTINE_PATH)
    
    spark.stop()

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, hour, dayofweek, when, lit
from loguru import logger
from config.spark_config import get_spark_session
import sys
import os

from src.utils.data_quality import validate_silver_transactions

def process_bronze_to_silver(
    spark: SparkSession, 
    bronze_path: str, 
    silver_path: str
) -> None:
    """
    Reads data from Bronze layer, performs cleaning and feature engineering,
    and writes to Silver layer in Delta format.
    
    Args:
        spark: Active SparkSession.
        bronze_path: Path to the Bronze Delta table.
        silver_path: Path to save the Silver Delta table.
    """
    logger.info(f"Starting Silver layer processing from {bronze_path}")
    
    try:
        # 1. Read from Bronze
        df_bronze = spark.read.format("delta").load(bronze_path)
        
        # 2. Basic Feature Engineering
        logger.info("Performing feature engineering...")
        df_silver = (
            df_bronze
            .withColumn("txn_hour", hour(col("timestamp")))
            .withColumn("txn_day_of_week", dayofweek(col("timestamp")))
            .withColumn("is_weekend", when(col("txn_day_of_week").isin([1, 7]), 1).otherwise(0))
            .withColumn("label", col("is_fraud").cast("int"))
            .fillna({"fraud_type": "none"})
        )
        
        # 3. Data Quality Gate (Great Expectations)
        validate_silver_transactions(df_silver)
        
        # 4. Write to Silver (Delta)
        # We can re-partition by ingestion_date or a more optimized column for analysis
        logger.info(f"Writing to Silver layer (Delta): {silver_path}")
        (
            df_silver.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("ingestion_date")
            .save(silver_path)
        )
        
        logger.success(f"Silver layer processing complete. Saved to {silver_path}")
        
    except Exception as e:
        logger.exception(f"Error during Silver layer processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    spark = get_spark_session(app_name="SilverTransformation")
    
    BRONZE_PATH = "data/bronze/transactions"
    SILVER_PATH = "data/silver/transactions"
    
    process_bronze_to_silver(spark, BRONZE_PATH, SILVER_PATH)
    
    spark.stop()

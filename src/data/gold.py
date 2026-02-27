from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, count, sum, avg, round
from loguru import logger
from config.spark_config import get_spark_session
import sys
import os

def create_gold_aggregations(
    spark: SparkSession, 
    silver_path: str, 
    gold_daily_path: str,
    gold_hourly_path: str
) -> None:
    """
    Reads from Silver layer and creates business-level aggregations.
    
    Args:
        spark: Active SparkSession.
        silver_path: Path to the Silver Delta table.
        gold_daily_path: Path to save daily fraud aggregations.
        gold_hourly_path: Path to save hourly transaction trends.
    """
    logger.info(f"Starting Gold layer aggregations from {silver_path}")
    
    try:
        # 1. Read from Silver
        df_silver = spark.read.format("delta").load(silver_path)
        
        # 2. Daily Fraud Summary
        logger.info("Calculating daily fraud metrics...")
        df_daily_fraud = (
            df_silver.groupBy("ingestion_date")
            .agg(
                count("transaction_id").alias("total_transactions"),
                sum("label").alias("fraud_cases"),
                round(avg("amount"), 2).alias("avg_transaction_amount"),
                sum("amount").alias("total_volume")
            )
            .withColumn("fraud_rate", round(col("fraud_cases") / col("total_transactions"), 4))
            .orderBy("ingestion_date")
        )
        
        # 3. Hourly Trend Analysis
        logger.info("Calculating hourly transaction trends...")
        df_hourly_trends = (
            df_silver.groupBy("txn_hour")
            .agg(
                count("transaction_id").alias("txn_count"),
                sum("label").alias("fraud_count"),
                round(avg("spending_deviation_score"), 4).alias("avg_deviation")
            )
            .orderBy("txn_hour")
        )
        
        # 4. Write to Gold (Delta)
        logger.info(f"Writing daily metrics to {gold_daily_path}")
        df_daily_fraud.write.format("delta").mode("overwrite").save(gold_daily_path)
        
        logger.info(f"Writing hourly trends to {gold_hourly_path}")
        df_hourly_trends.write.format("delta").mode("overwrite").save(gold_hourly_path)
        
        logger.success("Gold layer aggregations complete.")
        
    except Exception as e:
        logger.exception(f"Error during Gold layer aggregation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    spark = get_spark_session(app_name="GoldAggregations")
    
    SILVER_PATH = "data/silver/transactions"
    GOLD_DAILY_PATH = "data/gold/daily_fraud_summary"
    GOLD_HOURLY_PATH = "data/gold/hourly_transaction_trends"
    
    create_gold_aggregations(spark, SILVER_PATH, GOLD_DAILY_PATH, GOLD_HOURLY_PATH)
    
    spark.stop()

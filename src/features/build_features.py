from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import col, count, avg, collect_set, size, unix_timestamp, lit
from loguru import logger
from config.spark_config import get_spark_session
import sys
import os

def build_advanced_features(
    spark: SparkSession, 
    silver_path: str, 
    feature_path: str
) -> None:
    """
    Calculates window-based behavioral features from Silver transactions.
    
    Args:
        spark: Active SparkSession.
        silver_path: Path to the Silver Delta table.
        feature_path: Path to save the final feature set.
    """
    logger.info(f"Building advanced features from {silver_path}")
    
    try:
        # 1. Read from Silver
        df = spark.read.format("delta").load(silver_path)
        
        # Define window specifications based on timestamp (seconds)
        # 1 hour = 3600 seconds
        # 6 hours = 21600 seconds
        # 24 hours = 86400 seconds
        
        # We need a column with unix timestamp for rangeBetween
        df_timestamped = df.withColumn("ts_unix", unix_timestamp(col("timestamp")))
        
        # Window for 1h velocity
        w_1h = (Window.partitionBy("sender_account")
                .orderBy("ts_unix")
                .rangeBetween(-3600, 0))
        
        # Window for 24h moving average
        w_24h = (Window.partitionBy("sender_account")
                 .orderBy("ts_unix")
                 .rangeBetween(-86400, 0))
        
        # Window for 6h merchant diversity
        w_6h = (Window.partitionBy("sender_account")
                .orderBy("ts_unix")
                .rangeBetween(-21600, 0))
        
        logger.info("Calculating behavioral window features...")
        
        df_features = (
            df_timestamped
            # 1. Velocity (Count in last hour)
            .withColumn("velocity_1h", count("transaction_id").over(w_1h))
            
            # 2. Avg Difference 24h
            .withColumn("avg_amount_24h", avg("amount").over(w_24h))
            .withColumn("avg_diff_24h", col("amount") - col("avg_amount_24h"))
            
            # 3. Merchant Diversity (Unique receivers in last 6h)
            .withColumn("merchant_diversity", size(collect_set("receiver_account").over(w_6h)))
            
            # Cleanup intermediate columns
            .drop("ts_unix", "avg_amount_24h")
        )
        
        # 3. Write Features to Delta
        logger.info(f"Writing features to {feature_path}")
        (
            df_features.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .partitionBy("ingestion_date")
            .save(feature_path)
        )
        
        logger.success(f"Feature engineering complete. Saved to {feature_path}")
        
    except Exception as e:
        logger.exception(f"Error during feature building: {e}")
        sys.exit(1)

if __name__ == "__main__":
    spark = get_spark_session(app_name="FeatureEngineering")
    
    SILVER_PATH = "data/silver/transactions"
    FEATURE_PATH = "data/features/transaction_behavioral"
    
    build_advanced_features(spark, SILVER_PATH, FEATURE_PATH)
    
    spark.stop()

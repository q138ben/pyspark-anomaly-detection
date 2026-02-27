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
    Calculates window-based behavioral features for both train and test sets.
    """
    for split in ["train", "test"]:
        input_path = f"{silver_path}/{split}"
        output_path = f"{feature_path}/{split}"
        
        logger.info(f"Building features for {split} set from {input_path}")
        
        # 1. Read from Silver split
        df = spark.read.format("delta").load(input_path)
        
        # Define window specifications
        df_timestamped = df.withColumn("ts_unix", unix_timestamp(col("timestamp")))
        
        w_1h = (Window.partitionBy("sender_account").orderBy("ts_unix").rangeBetween(-3600, 0))
        w_24h = (Window.partitionBy("sender_account").orderBy("ts_unix").rangeBetween(-86400, 0))
        w_6h = (Window.partitionBy("sender_account").orderBy("ts_unix").rangeBetween(-21600, 0))
        
        # 2. Calculate features
        df_features = (
            df_timestamped
            .withColumn("velocity_1h", count("transaction_id").over(w_1h))
            .withColumn("avg_amount_24h", avg("amount").over(w_24h))
            .withColumn("avg_diff_24h", col("amount") - col("avg_amount_24h"))
            .withColumn("merchant_diversity", size(collect_set("receiver_account").over(w_6h)))
            .drop("ts_unix", "avg_amount_24h")
        )
        
        # 3. Write Features to Delta
        logger.info(f"Writing {split} features to {output_path}")
        (
            df_features.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(output_path)
        )
    
    logger.success("Feature engineering for all splits complete.")

if __name__ == "__main__":
    spark = get_spark_session(app_name="FeatureEngineering")
    
    SILVER_PATH = "data/silver/transactions"
    FEATURE_PATH = "data/features/transaction_behavioral"
    
    build_advanced_features(spark, SILVER_PATH, FEATURE_PATH)
    
    spark.stop()

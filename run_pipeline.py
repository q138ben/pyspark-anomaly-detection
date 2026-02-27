from config.spark_config import get_spark_session
from src.data.ingestion_pyspark import ingest_to_bronze
from src.data.silver import process_bronze_to_silver
from src.features.build_features import build_advanced_features
from src.data.gold import create_gold_aggregations
from src.models.predict import run_batch_inference
from loguru import logger
import mlflow
import sys
import os

def run_full_pipeline():
    """
    Orchestrates the entire Financial Anomaly Pipeline:
    Ingest -> Silver (Split) -> Features -> Gold Aggs -> Inference
    """
    logger.info("🚀 Initializing End-to-End Orchestrator")
    
    spark = get_spark_session(app_name="FraudDetectionOrchestrator")
    
    # Centralized Path Management
    RAW_PATH = "data/raw/financial_fraud_detection_dataset.csv"
    BRONZE_PATH = "data/bronze/transactions"
    QUARANTINE_PATH = "data/bronze/quarantine"
    SILVER_PATH = "data/silver/transactions"
    FEATURE_PATH = "data/features/transaction_behavioral"
    GOLD_DAILY_PATH = "data/gold/daily_fraud_summary"
    GOLD_HOURLY_PATH = "data/gold/hourly_transaction_trends"
    PRED_OUTPUT_PATH = "data/gold/fraud_predictions"
    
    try:
        # Step 1: Raw to Bronze
        logger.info("--- 📥 Step 1: Ingestion (Raw -> Bronze) ---")
        ingest_to_bronze(spark, RAW_PATH, BRONZE_PATH, QUARANTINE_PATH)
        
        # Step 2: Bronze to Silver (Feature Engineering & Splitting)
        logger.info("--- 🛠️ Step 2: Silver Transformation (Bronze -> Silver) ---")
        process_bronze_to_silver(spark, BRONZE_PATH, SILVER_PATH)
        
        # Step 3: Silver to Features (Window Behavioral Features)
        logger.info("--- 🧬 Step 3: Feature Building (Silver -> Features) ---")
        build_advanced_features(spark, SILVER_PATH, FEATURE_PATH)
        
        # Step 4: Silver to Gold Aggregations (Business Metrics)
        logger.info("--- 📊 Step 4: Gold Aggregations (Silver -> Gold) ---")
        create_gold_aggregations(spark, SILVER_PATH, GOLD_DAILY_PATH, GOLD_HOURLY_PATH)
        
        # Step 5: Model Inference (Features -> Predictions)
        logger.info("--- 🤖 Step 5: Batch Inference ---")
        
        # Programmatically find the latest model from the Split-based experiment
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("FinancialFraudDetection_V2")
        
        if experiment:
            runs = client.search_runs(
                experiment.experiment_id, 
                order_by=["attributes.start_time DESC"], 
                max_results=1
            )
            if runs:
                latest_run_id = runs[0].info.run_id
                MODEL_URI = f"runs:/{latest_run_id}/best_fraud_rf_model"
                run_batch_inference(spark, MODEL_URI, f"{FEATURE_PATH}/test", PRED_OUTPUT_PATH)
            else:
                logger.warning("⚠️ No MLflow runs found for inference. Skipping Step 5.")
        else:
            logger.warning("⚠️ Experiment 'FinancialFraudDetection_V2' not found. Skipping Step 5.")
            
        logger.success("✅ End-to-End Pipeline Execution Successful!")
        
    except Exception as e:
        logger.exception(f"❌ Pipeline failed at some stage: {e}")
        sys.exit(1)
    finally:
        logger.info("Stopping Spark Session...")
        spark.stop()

if __name__ == "__main__":
    run_full_pipeline()

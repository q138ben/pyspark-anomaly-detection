from pyspark.sql import SparkSession, DataFrame
from loguru import logger
import mlflow.spark
from config.spark_config import get_spark_session
import sys
import os

def run_batch_inference(
    spark: SparkSession,
    model_uri: str,
    feature_test_path: str,
    output_path: str
) -> None:
    """
    Loads a trained model from MLflow and generates predictions on the test set.
    """
    logger.info(f"Loading model from: {model_uri}")
    try:
        # 1. Load the model
        model = mlflow.spark.load_model(model_uri)
        
        # 2. Load test features
        logger.info(f"Loading test features from: {feature_test_path}")
        test_df = spark.read.format("delta").load(feature_test_path)
        
        # Handle nulls in behavioral features
        feature_cols = ["velocity_1h", "avg_diff_24h", "merchant_diversity"]
        test_df = test_df.fillna(0, subset=feature_cols)
        
        # 3. Generate Predictions
        logger.info("Generating predictions...")
        predictions = model.transform(test_df)
        
        # 4. Save results to Gold layer
        # Selecting key columns for the final output
        final_results = predictions.select(
            "transaction_id",
            "timestamp",
            "sender_account",
            "amount",
            "label", # Ground truth (if available)
            "prediction", # Binary prediction (0 or 1)
            "probability"  # Confidence score
        )
        
        logger.info(f"Saving predictions to: {output_path}")
        (
            final_results.write
            .format("delta")
            .mode("overwrite")
            .save(output_path)
        )
        
        logger.success("Batch inference complete.")
        
    except Exception as e:
        logger.exception(f"Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    spark = get_spark_session(app_name="FraudInference")
    
    # In a production setting, these would be passed via environment variables or CLI args
    # We'll use the relative path to the latest run in mlruns for this demo
    # Note: Replace with actual run ID if needed, or use 'models:/model_name/production' if registered
    
    # Finding the latest run ID programmatically
    import mlflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("FinancialFraudDetection_V2")
    runs = client.search_runs(experiment.experiment_id, order_by=["attributes.start_time DESC"], max_results=1)
    
    if not runs:
        logger.error("No trained models found in MLflow. Please run training first.")
        sys.exit(1)
        
    latest_run_id = runs[0].info.run_id
    MODEL_URI = f"runs:/{latest_run_id}/best_fraud_rf_model"
    FEATURE_TEST_PATH = "data/features/transaction_behavioral/test"
    OUTPUT_PATH = "data/gold/fraud_predictions"
    
    run_batch_inference(spark, MODEL_URI, FEATURE_TEST_PATH, OUTPUT_PATH)
    
    spark.stop()

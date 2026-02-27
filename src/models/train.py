from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from loguru import logger
import mlflow
import mlflow.spark
from config.spark_config import get_spark_session
import sys
import os

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

def train_fraud_model(
    spark: SparkSession, 
    feature_path: str
) -> None:
    """
    Trains a Random Forest classifier with Cross-Validation and tracks with MLflow.
    """
    logger.info(f"Starting model training with Cross-Validation from {feature_path}")
    
    # 1. Load and Sample Data
    df = spark.read.format("delta").load(feature_path)
    df_clean = df.fillna(0, subset=["velocity_1h", "avg_diff_24h", "merchant_diversity"])
    
    logger.info("Sampling dataset (5%) for faster Cross-Validation...")
    df_sampled = df_clean.sampleBy("label", fractions={0: 0.05, 1: 1.0}, seed=42)
    train_df, test_df = df_sampled.randomSplit([0.8, 0.2], seed=42)
    
    # 2. Build Pipeline
    input_cols = ["amount", "time_since_last_transaction", "spending_deviation_score", 
                  "velocity_score", "geo_anomaly_score", "velocity_1h", "avg_diff_24h", "merchant_diversity"]
    
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    
    # 3. Setup ParamGrid and CrossValidator
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.numTrees, [20, 50])
                 .addGrid(rf.maxDepth, [5, 10])
                 .build())
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,
        seed=42
    )
    
    # 4. Train and Log
    mlflow.set_experiment("FinancialFraudDetection")
    
    with mlflow.start_run(run_name="RandomForest_CrossVal"):
        logger.info("Starting Cross-Validation fit...")
        cv_model = cv.fit(train_df)
        
        # Best model results
        best_pipeline = cv_model.bestModel
        best_rf = best_pipeline.stages[-1]
        
        mlflow.log_param("best_num_trees", best_rf.getNumTrees)
        mlflow.log_param("best_max_depth", best_rf.getMaxDepth())
        
        # Log the best model
        mlflow.spark.log_model(best_pipeline, "best_fraud_rf_model")
        
        # Final Evaluation
        predictions = cv_model.transform(test_df)
        roc_auc = evaluator.evaluate(predictions)
        mlflow.log_metric("final_roc_auc", roc_auc)
        
        logger.success(f"Cross-Validation complete. Best Test ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    spark = get_spark_session(app_name="FraudModelTrainingCV")
    
    FEATURE_PATH = "data/features/transaction_behavioral"
    
    train_fraud_model(spark, FEATURE_PATH)
    
    spark.stop()

from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from loguru import logger
import mlflow
import mlflow.spark
from config.spark_config import get_spark_session
import sys
import os

def train_fraud_model(
    spark: SparkSession, 
    feature_base_path: str
) -> None:
    """
    Trains a Random Forest classifier using pre-split chronological features.
    """
    logger.info(f"Starting model training from {feature_base_path}")
    
    # 1. Load Pre-split Features
    train_path = f"{feature_base_path}/train"
    test_path = f"{feature_base_path}/test"
    
    train_df = spark.read.format("delta").load(train_path)
    test_df = spark.read.format("delta").load(test_path)
    
    # Handle possible nulls (using 0 for behavioral features)
    feature_cols = ["velocity_1h", "avg_diff_24h", "merchant_diversity"]
    train_df = train_df.fillna(0, subset=feature_cols)
    test_df = test_df.fillna(0, subset=feature_cols)
    
    # Stratified Sampling for local development efficiency (Optional)
    logger.info("Sampling training data (10%) for local efficiency...")
    train_sampled = train_df.sampleBy("label", fractions={0: 0.1, 1: 1.0}, seed=42)
    
    # 2. Build Pipeline
    input_cols = [
        "amount", "time_since_last_transaction", "spending_deviation_score", 
        "velocity_score", "geo_anomaly_score", "velocity_1h", 
        "avg_diff_24h", "merchant_diversity"
    ]
    
    assembler = VectorAssembler(inputCols=input_cols, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withStd=True, withMean=True)
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed=42)
    
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    
    # 3. Setup Hyperparameter Tuning
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
    
    # 4. Train and Log with MLflow
    mlflow.set_experiment("FinancialFraudDetection_V2") # V2 for split-based
    
    with mlflow.start_run(run_name="RandomForest_ChronologicalSplit"):
        logger.info("Fitting CrossValidator on training set...")
        cv_model = cv.fit(train_sampled)
        
        best_pipeline = cv_model.bestModel
        best_rf = best_pipeline.stages[-1]
        
        # Log Best Params
        mlflow.log_param("best_num_trees", best_rf.getNumTrees)
        mlflow.log_param("best_max_depth", best_rf.getMaxDepth())
        mlflow.log_param("split_strategy", "chronological_2023-10-20")
        
        # Log Model
        mlflow.spark.log_model(best_pipeline, "best_fraud_rf_model")
        
        # 5. Final Evaluation on UNSEEN Test Set
        logger.info("Evaluating on unseen test set...")
        predictions = cv_model.transform(test_df)
        roc_auc = evaluator.evaluate(predictions)
        mlflow.log_metric("test_roc_auc", roc_auc)
        
        logger.success(f"Training complete. Unseen Test ROC-AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    spark = get_spark_session(app_name="FraudModelTrainingV2")
    
    FEATURE_BASE_PATH = "data/features/transaction_behavioral"
    
    train_fraud_model(spark, FEATURE_BASE_PATH)
    
    spark.stop()

from pyspark.sql import SparkSession
from loguru import logger
import sys

def get_spark_session(app_name: str = "FinancialAnomalyPipeline", master: str = "local[*]") -> SparkSession:
    """
    Initializes a production-ready SparkSession with optimized configurations.
    
    Args:
        app_name (str): Name of the Spark application.
        master (str): Spark master URL. Defaults to 'local[*]' for development.
        
    Returns:
        SparkSession: A configured Spark session.
    """
    logger.info(f"Initializing Spark Session: {app_name}")
    
    try:
        spark = (
            SparkSession.builder
            .appName(app_name)
            .master(master)
            # Adaptive Query Execution (AQE)
            .config("spark.sql.adaptive.enabled", "true")
            # Kryo Serializer for efficiency
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            # Standardizing timezone
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate()
        )
        
        logger.success(f"Spark Session '{app_name}' initialized successfully (Version: {spark.version})")
        return spark
    except Exception as e:
        logger.exception(f"Failed to initialize Spark Session: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Smoke test for initialization
    spark = get_spark_session()
    spark.stop()

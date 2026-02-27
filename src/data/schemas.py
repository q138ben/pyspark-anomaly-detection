from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    TimestampType, BooleanType, IntegerType
)
from typing import Final

# Complete production-grade schema for financial transactions based on raw CSV metadata.
TRANSACTION_SCHEMA: Final[StructType] = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("sender_account", StringType(), False),
    StructField("receiver_account", StringType(), False),
    StructField("amount", DoubleType(), False),
    StructField("transaction_type", StringType(), False),
    StructField("merchant_category", StringType(), False),
    StructField("location", StringType(), False),
    StructField("device_used", StringType(), False),
    StructField("is_fraud", BooleanType(), False),
    StructField("fraud_type", StringType(), True), # Often null for non-fraud
    StructField("time_since_last_transaction", DoubleType(), True),
    StructField("spending_deviation_score", DoubleType(), True),
    StructField("velocity_score", IntegerType(), True),
    StructField("geo_anomaly_score", DoubleType(), True),
    StructField("payment_channel", StringType(), True),
    StructField("ip_address", StringType(), True),
    StructField("device_hash", StringType(), True),
    StructField("_corrupt_record", StringType(), True), # Quarantine for malformed rows
])

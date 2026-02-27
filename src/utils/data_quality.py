import great_expectations as gx
from pyspark.sql import DataFrame
from loguru import logger
import sys

def validate_silver_transactions(df: DataFrame) -> None:
    """
    Validates the Silver layer transactions using Great Expectations.
    """
    logger.info("Running Great Expectations validation on Silver layer...")
    
    try:
        # 1. Initialize ephemeral GX context
        context = gx.get_context()
        
        # 2. Create the Expectation Suite
        suite_name = "silver_transaction_suite"
        suite = context.suites.add(gx.ExpectationSuite(name=suite_name))
        
        # 3. Get Validator
        validator = context.get_validator(
            batch_request=context.data_sources.add_spark(name="my_spark_datasource")
            .add_dataframe_asset(name="my_df_asset")
            .build_batch_request(options={"dataframe": df}),
            expectation_suite_name=suite_name
        )
        
        # 4. Add expectations
        validator.expect_column_to_exist("transaction_id")
        validator.expect_column_to_exist("amount")
        validator.expect_column_to_exist("label")
        validator.expect_column_values_to_not_be_null("transaction_id")
        validator.expect_column_values_to_not_be_null("amount")
        validator.expect_column_values_to_be_between("amount", min_value=0)
        validator.expect_column_values_to_be_between("txn_hour", min_value=0, max_value=23)
        validator.expect_column_values_to_be_in_set("label", [0, 1])
        
        # 5. Execute validation
        validation_result = validator.validate()
        
        if not validation_result.success:
            logger.error("Great Expectations validation failed for Silver layer!")
            raise ValueError("Data Quality Gate failed: Silver layer validation unsuccessful.")
            
        logger.success("Great Expectations validation passed for Silver layer.")
        
    except Exception as e:
        logger.exception(f"Error during data validation: {e}")
        raise

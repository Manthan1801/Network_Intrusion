import os
import sys
import pandas as pd
from NetworkIntrusion.logging.logger import logging
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.entity.config_entity import DataValidationConfig
from NetworkIntrusion.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from NetworkIntrusion.utils.main_utils import read_yaml_file, write_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(self.data_validation_config.schema_file_path)
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            dataframe_columns = dataframe.columns
            schema_columns = list(self._schema_config["columns"].keys())

            missing_columns = []
            for column in schema_columns:
                if column not in dataframe_columns:
                    missing_columns.append(column)

            if len(missing_columns) > 0:
                logging.warning(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting Data Validation Phase")
            
            error_message = ""
            train_df = pd.read_parquet(self.data_ingestion_artifact.trained_file_path)
            test_df = pd.read_parquet(self.data_ingestion_artifact.test_file_path)

            # Strip whitespace from column names (EDA Recommendation 1)
            train_df.columns = train_df.columns.str.strip()
            test_df.columns = test_df.columns.str.strip()

            status_train_cols = self.validate_number_of_columns(dataframe=train_df)
            status_test_cols = self.validate_number_of_columns(dataframe=test_df)
            if not status_train_cols or not status_test_cols:
                error_message += "Train or test dataframe does not contain exact number of columns. "

            status_train_exists = self.is_column_exist(dataframe=train_df)
            status_test_exists = self.is_column_exist(dataframe=test_df)
            if not status_train_exists or not status_test_exists:
                error_message += "Train or test dataframe does not contain all required columns. "

            validation_status = len(error_message) == 0

            # Generate validation report
            validation_report = {
                "validation_status": validation_status,
                "error_message": error_message if not validation_status else "All columns are present and valid."
            }

            write_yaml_file(
                file_path=self.data_validation_config.validation_report_file_path,
                content=validation_report,
                replace=True
            )

            # Save dataframes to valid or invalid directories based on status
            if validation_status:
                os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
                train_df.to_parquet(self.data_validation_config.valid_train_file_path, index=False)
                test_df.to_parquet(self.data_validation_config.valid_test_file_path, index=False)
                logging.info(f"Data is valid. Saved to {self.data_validation_config.valid_data_dir}")
            else:
                os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)
                train_df.to_parquet(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_parquet(self.data_validation_config.invalid_test_file_path, index=False)
                logging.error("Data validation failed. Saved to invalid directory.")

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            logging.info("Data Validation Phase completed.")
            return data_validation_artifact
            
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

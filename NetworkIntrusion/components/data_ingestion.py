import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from NetworkIntrusion.entity.config_entity import DataIngestionConfig
from NetworkIntrusion.entity.artifact_entity import DataIngestionArtifact
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.logging.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion Phase")
            dataset_path = self.data_ingestion_config.dataset_path

            if not os.path.exists(dataset_path):
                raise Exception(f"Dataset not found at {dataset_path}")

            logging.info(f"Reading dataset from {dataset_path}")
            # Loading data as pandas dataframe
            df = pd.read_parquet(dataset_path)
            logging.info(f"Dataset read successfully. Shape: {df.shape}")

            # Create directories for feature store and ingested data
            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)
            
            # Save raw dataset to feature store
            df.to_parquet(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Raw dataset saved to feature store at {self.data_ingestion_config.feature_store_file_path}")

            # Train test split
            logging.info("Performing train test split")
            train_set, test_set = train_test_split(
                df, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=42
            )
            logging.info(f"Train test split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

            # Save train and test sets
            train_set.to_parquet(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_parquet(self.data_ingestion_config.testing_file_path, index=False)
            logging.info("Train and test datasets saved to artifacts")

            # Create artifact entity
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )
            logging.info("Data Ingestion Phase completed successfully")

            return data_ingestion_artifact

        except Exception as e:
            raise NetworkIntrusionException(e, sys)
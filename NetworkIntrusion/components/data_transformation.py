import os
import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
from imblearn.under_sampling import RandomUnderSampler
from NetworkIntrusion.logging.logger import logging
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.entity.config_entity import DataTransformationConfig
from NetworkIntrusion.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from NetworkIntrusion.utils.main_utils import save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            # Constant columns to remove as per EDA
            self.drop_columns = [
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 
                'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 
                'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
            ]
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("Creating data transformer object (Pipeline)")
            from sklearn.preprocessing import RobustScaler
            # Imputer to handle NaNs (which will also cover infs once replaced)
            preprocessor = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler())
                ]
            )
            return preprocessor
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation Phase")
            
            # Read validated data
            train_df = pd.read_parquet(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_parquet(self.data_validation_artifact.valid_test_file_path)

            logging.info("Removing duplicate rows from train and test data")
            train_df.drop_duplicates(inplace=True)
            test_df.drop_duplicates(inplace=True)

            logging.info(f"Dropping constant columns: {self.drop_columns}")
            # Ensure columns exist before dropping (prevent errors if already dropped)
            cols_to_drop = [col for col in self.drop_columns if col in train_df.columns]
            train_df.drop(columns=cols_to_drop, inplace=True)
            test_df.drop(columns=cols_to_drop, inplace=True)

            # Split into input features and target
            target_column_name = "Label"
            
            X_train = train_df.drop(columns=[target_column_name], axis=1)
            y_train = train_df[target_column_name]

            X_test = test_df.drop(columns=[target_column_name], axis=1)
            y_test = test_df[target_column_name]

            logging.info("Replacing infinite values with NaN")
            X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
            X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

            logging.info("Encoding target labels")
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)

            # Apply preprocessing
            logging.info("Applying preprocessing (imputation + scaling)")
            preprocessor_obj = self.get_data_transformer_object()
            
            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            # Combine X and y
            train_arr = np.c_[X_train_transformed, y_train_encoded]
            test_arr = np.c_[X_test_transformed, y_test_encoded]

            # Save transformed arrays
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            logging.info("Saved transformed train and test data.")

            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_obj
            )
            logging.info(f"Saved preprocessor object to {self.data_transformation_config.transformed_object_file_path}")

            # Save label encoder
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=label_encoder
            )
            logging.info(f"Saved label encoder to {self.data_transformation_config.label_encoder_file_path}")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                label_encoder_file_path=self.data_transformation_config.label_encoder_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data Transformation Phase completed successfully.")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkIntrusionException(e, sys)

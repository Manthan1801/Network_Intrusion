import os
import sys
import pandas as pd
import numpy as np
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.logging.logger import logging
from NetworkIntrusion.utils.main_utils import load_object

class NetworkIntrusionModel:
    def __init__(self, preprocessor_path: str, model_path: str, label_encoder_path: str):
        try:
            self.preprocessor_path = preprocessor_path
            self.model_path = model_path
            self.label_encoder_path = label_encoder_path
            self.preprocessor = None
            self.model = None
            self.label_encoder = None
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def _load_objects(self):
        try:
            if self.preprocessor is None:
                logging.info(f"Loading preprocessor from {self.preprocessor_path}")
                self.preprocessor = load_object(file_path=self.preprocessor_path)
            if self.model is None:
                logging.info(f"Loading model from {self.model_path}")
                self.model = load_object(file_path=self.model_path)
            if self.label_encoder is None:
                logging.info(f"Loading label encoder from {self.label_encoder_path}")
                self.label_encoder = load_object(file_path=self.label_encoder_path)
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> dict:
        try:
            logging.info("Initializing prediction pipeline")
            self._load_objects()
            dataframe.columns = dataframe.columns.str.strip()

            # Align features with training: drop constant columns if they exist
            drop_columns = [
                'Bwd PSH Flags', 'Bwd URG Flags', 'Fwd Avg Bytes/Bulk', 
                'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 
                'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate'
            ]
            cols_to_drop = [col for col in drop_columns if col in dataframe.columns]
            if cols_to_drop:
                dataframe = dataframe.drop(columns=cols_to_drop)

            # Replace infinite values with NaN (SimpleImputer handles NaN)
            dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Transform input data
            logging.info("Transforming input features")
            transformed_features = self.preprocessor.transform(dataframe)

            # Predict probabilities
            logging.info("Making predictions with the trained model")
            
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(transformed_features)
                max_probs = probabilities.max(axis=1) # Get the confidence of the predicted class
                predictions = probabilities.argmax(axis=1)
            else:
                predictions = self.model.predict(transformed_features)
                max_probs = [1.0] * len(predictions) # Default to 100% confidence if predict_proba is missing

            logging.info("Mapping numerical predictions to human-readable attack classes")
            readable_predictions = self.label_encoder.inverse_transform(predictions.astype(int))

            return {
                "predictions": readable_predictions.tolist(),
                "confidences": [float(p * 100) for p in max_probs] # Convert to percentage
            }
            
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

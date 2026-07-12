import os
import pandas as pd
from typing import Dict, Any, List
from NetworkIntrusion.utils.ml_utils.model.estimator import NetworkIntrusionModel

class PredictionPipeline:
    def __init__(self):
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Automatically finds the latest artifacts and initializes the model."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        artifacts_dir = os.path.join(base_dir, "artifacts")
        
        if not os.path.exists(artifacts_dir):
            raise Exception("Artifacts directory not found. Please train the model first.")
            
        latest_run = sorted(os.listdir(artifacts_dir))[-1]
        run_dir = os.path.join(artifacts_dir, latest_run)
        
        preprocessor_path = os.path.join(run_dir, "data_transformation", "transformed_object", "preprocessing.pkl")
        model_path = os.path.join(run_dir, "model_trainer", "trained_model", "model.pkl")
        label_encoder_path = os.path.join(run_dir, "data_transformation", "transformed_object", "label_encoder.pkl")
        
        self.model = NetworkIntrusionModel(
            preprocessor_path=preprocessor_path,
            model_path=model_path,
            label_encoder_path=label_encoder_path
        )

    def predict_single(self, features: Dict[str, Any]) -> dict:
        """Predicts a single network flow."""
        # Convert single dict to DataFrame
        df = pd.DataFrame([features])
        result = self.model.predict(df)
        
        return {
            "prediction": result["predictions"][0],
            "confidence": result["confidences"][0]
        }
        
    def predict_batch(self, features_list: List[Dict[str, Any]]) -> List[dict]:
        """Predicts a batch of network flows."""
        df = pd.DataFrame(features_list)
        results = self.model.predict(df)
        
        batch_results = []
        for i in range(len(results["predictions"])):
            batch_results.append({
                "prediction": results["predictions"][i],
                "confidence": results["confidences"][i]
            })
            
        return batch_results

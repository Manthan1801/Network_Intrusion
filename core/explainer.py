# pyrefly: ignore [missing-import]
import shap
import pandas as pd
import numpy as np
import logging

class ThreatExplainer:
    def __init__(self, model, feature_names):
        """
        Initializes SHAP TreeExplainer. TreeExplainer is extremely fast for LightGBM/XGBoost.
        """
        self.model = model
        self.feature_names = feature_names
        try:
            # We use TreeExplainer for tree-based models like LightGBM
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logging.error(f"Failed to initialize SHAP explainer: {e}")
            self.explainer = None

    def explain_attack(self, transformed_features: np.ndarray) -> str:
        """
        Generates explanation only for malicious predictions.
        Returns the top contributing feature name.
        """
        if self.explainer is None:
            return "Explainability Unavailable"
            
        try:
            # Calculate SHAP values for the given instance
            shap_values = self.explainer.shap_values(transformed_features)
            
            # For multi-class, shap_values might be a list. We take the max or absolute sum.
            # Simplify by taking the absolute values across all classes/features
            if isinstance(shap_values, list):
                # Aggregate across classes by taking max absolute impact
                mean_abs_shap = np.max(np.abs(np.array(shap_values)), axis=0).flatten()
            else:
                mean_abs_shap = np.abs(shap_values).flatten()
                
            # Find the index of the highest contributing feature
            top_feature_idx = np.argmax(mean_abs_shap)
            
            # Map index to feature name
            if self.feature_names and top_feature_idx < len(self.feature_names):
                return self.feature_names[top_feature_idx]
            else:
                return f"Feature_{top_feature_idx}"
                
        except Exception as e:
            logging.error(f"Error generating SHAP explanation: {e}")
            return "Error during explanation"

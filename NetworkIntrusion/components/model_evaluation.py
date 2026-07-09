import os
import sys
import numpy as np
# pyrefly: ignore [missing-import]
import mlflow
# pyrefly: ignore [missing-import]
import mlflow.sklearn
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.logging.logger import logging
from NetworkIntrusion.entity.config_entity import ModelEvaluationConfig
from NetworkIntrusion.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise NetworkIntrusionException(e, sys)
            
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation Phase")
            
            # Load Data
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            test_arr = np.load(test_file_path)
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            # Load Model and Label Encoder
            with open(self.model_trainer_artifact.trained_model_file_path, 'rb') as f:
                model = pickle.load(f)
                
            with open(self.data_transformation_artifact.label_encoder_file_path, 'rb') as f:
                label_encoder = pickle.load(f)
            
            logging.info("Making predictions on test dataset")
            y_pred = model.predict(X_test)
            
            # Inverse transform to get original string labels
            y_test_original = label_encoder.inverse_transform(y_test.astype(int))
            y_pred_original = label_encoder.inverse_transform(y_pred.astype(int))
            
            logging.info("Calculating metrics and confusion matrix")
            # Calculate metrics
            accuracy = accuracy_score(y_test_original, y_pred_original)
            f1_weighted = f1_score(y_test_original, y_pred_original, average='weighted')
            
            # Generate classification report dict for per-class precision/recall/f1
            class_report = classification_report(y_test_original, y_pred_original, output_dict=True)
            
            # Confusion Matrix
            cm = confusion_matrix(y_test_original, y_pred_original, labels=label_encoder.classes_)
            
            # Plot Confusion Matrix
            os.makedirs(self.model_evaluation_config.model_evaluation_dir, exist_ok=True)
            plt.figure(figsize=(15, 10))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=label_encoder.classes_, 
                        yticklabels=label_encoder.classes_)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.model_evaluation_config.confusion_matrix_path)
            plt.close()
            logging.info(f"Saved confusion matrix plot to {self.model_evaluation_config.confusion_matrix_path}")

            # MLflow tracking
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            if "MLFLOW_TRACKING_URI" in os.environ:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
                logging.info(f"Using MLflow tracking URI from environment: {os.environ['MLFLOW_TRACKING_URI']}")
            
            with mlflow.start_run():
                logging.info("Logging metrics to MLflow")
                
                # Log global metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("f1_weighted", f1_weighted)
                
                # Log per-class metrics (only for specific important minor classes to avoid clutter, or log all)
                for class_name in label_encoder.classes_:
                    if class_name in class_report:
                        mlflow.log_metric(f"f1_{class_name}", class_report[class_name]['f1-score'])
                        mlflow.log_metric(f"recall_{class_name}", class_report[class_name]['recall'])
                        mlflow.log_metric(f"precision_{class_name}", class_report[class_name]['precision'])

                # Log confusion matrix artifact
                mlflow.log_artifact(self.model_evaluation_config.confusion_matrix_path)
                
                # Log model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="NetworkIntrusionModel")
                else:
                    mlflow.sklearn.log_model(model, "model")
                    
                logging.info("Successfully logged to MLflow")

            # Creating Evaluation Artifact
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=True,
                improved_accuracy=accuracy, # Simply storing accuracy for now
                best_model_path=self.model_trainer_artifact.trained_model_file_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                train_model_metric_artifact=self.model_trainer_artifact.train_metric_artifact,
                best_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact
            )
            
            logging.info("Model Evaluation Component completed successfully")
            return model_evaluation_artifact

        except Exception as e:
            raise NetworkIntrusionException(e, sys)

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from NetworkIntrusion.exception.exception import NetworkIntrusionException
from NetworkIntrusion.logging.logger import logging
from NetworkIntrusion.entity.config_entity import ModelTrainerConfig
from NetworkIntrusion.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from NetworkIntrusion.utils.main_utils import save_object
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
# pyrefly: ignore [missing-import]
from xgboost import XGBClassifier
# pyrefly: ignore [missing-import]
from lightgbm import LGBMClassifier
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def get_classification_metric(self, y_true, y_pred) -> ClassificationMetricArtifact:
        try:
            f1 = f1_score(y_true, y_pred, average="weighted")
            precision = precision_score(y_true, y_pred, average="weighted")
            recall = recall_score(y_true, y_pred, average="weighted")
            accuracy = accuracy_score(y_true, y_pred)

            return ClassificationMetricArtifact(
                f1_score=f1,
                precision_score=precision,
                recall_score=recall,
                accuracy_score=accuracy
            )
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}
            for i in range(len(list(models))):
                model_name = list(models.keys())[i]
                model = list(models.values())[i]
                param = params[model_name]

                logging.info(f"Training model: {model_name}")
                rs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param,
                    n_iter=10,
                    cv=3,
                    verbose=3,
                    n_jobs=-1,
                    scoring="f1_weighted",
                    random_state=42
                )
                rs.fit(X_train, y_train)

                model.set_params(**rs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                test_model_score = accuracy_score(y_test, y_test_pred)

                report[model_name] = {
                    "test_score": test_model_score,
                    "best_params": rs.best_params_
                }

            return report
        except Exception as e:
            raise NetworkIntrusionException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Trainer Component")

            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading transformed train and test data")
            train_arr = np.load(train_file_path)
            test_arr = np.load(test_file_path)

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                "Random Forest": RandomForestClassifier(verbose=1,n_jobs=-1),
                "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                "LightGBM": LGBMClassifier()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [None, 10, 20]
                },
                "XGBoost": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [3, 6, 10]
                },
                "LightGBM": {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'max_depth': [-1, 10, 20]
                }
            }

            logging.info("Creating a 10% stratified sample of the training data for hyperparameter tuning to reduce time")
            X_train_sample, _, y_train_sample, _ = train_test_split(
                X_train, y_train, train_size=0.1, stratify=y_train, random_state=42
            )

            logging.info("Evaluating models with RandomizedSearchCV on the subset")
            model_report: dict = self.evaluate_models(
                X_train=X_train_sample, y_train=y_train_sample, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # Get best model score from dict
            best_model_score = max([report["test_score"] for report in model_report.values()])

            # Get best model name from dict
            best_model_name = list(model_report.keys())[
                [report["test_score"] for report in model_report.values()].index(best_model_score)
            ]

            best_model = models[best_model_name]
            best_model_params = model_report[best_model_name]["best_params"]
            
            logging.info(f"Best model found: {best_model_name} with sample Accuracy Score: {best_model_score}")
            logging.info(f"Best parameters: {best_model_params}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found. Best accuracy is {best_model_score} which is less than the expected accuracy threshold of {self.model_trainer_config.expected_accuracy}."
                )

            logging.info(f"Retraining the best model ({best_model_name}) on the FULL training dataset")
            best_model.set_params(**best_model_params)
            best_model.fit(X_train, y_train)

            # Predict to get final metrics
            y_train_pred = best_model.predict(X_train)
            train_metric = self.get_classification_metric(y_true=y_train, y_pred=y_train_pred)

            y_test_pred = best_model.predict(X_test)
            test_metric = self.get_classification_metric(y_true=y_test, y_pred=y_test_pred)

            # Check for overfitting/underfitting
            diff = abs(train_metric.accuracy_score - test_metric.accuracy_score)
            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                logging.warning(
                    f"Model might be overfitting or underfitting. Train Acc: {train_metric.accuracy_score}, Test Acc: {test_metric.accuracy_score}, Diff: {diff}"
                )

            # Save the best model
            logging.info(f"Saving best model to {self.model_trainer_config.trained_model_file_path}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            
            logging.info("Model Trainer Component completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkIntrusionException(e, sys)

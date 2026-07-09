import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
load_dotenv()

from NetworkIntrusion.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
from NetworkIntrusion.components.data_ingestion import DataIngestion
from NetworkIntrusion.components.data_validation import DataValidation
from NetworkIntrusion.components.data_transformation import DataTransformation
from NetworkIntrusion.components.model_trainer import ModelTrainer
from NetworkIntrusion.components.model_evaluation import ModelEvaluation
import sys
from NetworkIntrusion.exception.exception import NetworkIntrusionException

def test_pipeline():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        
        # Data Ingestion
        print("\n--- Data Ingestion ---")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(f"Ingestion Successful: Train={ingestion_artifact.trained_file_path}, Test={ingestion_artifact.test_file_path}")

        # Data Validation
        print("\n--- Data Validation ---")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(ingestion_artifact, data_validation_config)
        validation_artifact = data_validation.initiate_data_validation()
        print(f"Validation Successful: Status={validation_artifact.validation_status}")

        if not validation_artifact.validation_status:
            print("Data Validation Failed. Halting pipeline.")
            return

        # Data Transformation
        print("\n--- Data Transformation ---")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(validation_artifact, data_transformation_config)
        transformation_artifact = data_transformation.initiate_data_transformation()
        print(f"Transformation Successful!")
        print(f"Transformed Train Data: {transformation_artifact.transformed_train_file_path}")
        print(f"Transformed Test Data: {transformation_artifact.transformed_test_file_path}")
        print(f"Preprocessor Object: {transformation_artifact.transformed_object_file_path}")

        # Model Trainer
        print("\n--- Model Trainer ---")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(transformation_artifact, model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        
        print(f"Model Training Successful!")
        print(f"Final Model File: {model_trainer_artifact.trained_model_file_path}")
        print(f"Test Accuracy: {model_trainer_artifact.test_metric_artifact.accuracy_score}")

        # Model Evaluation
        print("\n--- Model Evaluation ---")
        model_eval_config = ModelEvaluationConfig(training_pipeline_config)
        model_eval = ModelEvaluation(model_eval_config, transformation_artifact, model_trainer_artifact)
        model_eval_artifact = model_eval.initiate_model_evaluation()
        print(f"Model Evaluation Successful! Metrics logged to MLflow.")

    except Exception as e:
        print(f"Exception Occurred: {e}")
        raise NetworkIntrusionException(e, sys)

if __name__ == "__main__":
    test_pipeline()

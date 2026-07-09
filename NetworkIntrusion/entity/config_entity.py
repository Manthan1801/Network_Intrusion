import os
from datetime import datetime

class TrainingPipelineConfig:
    def __init__(self):
        timestamp = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = "NetworkIntrusion"
        self.artifact_dir: str = os.path.join("artifacts", timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_ingestion"
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, "feature_store", "CIC_IDS2017.parquet"
        )
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir, "ingested", "train.parquet"
        )
        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir, "ingested", "test.parquet"
        )
        self.train_test_split_ratio: float = 0.2
        # downloaded dataset path
        self.dataset_path: str = os.path.join("Network_intrusion_data", "CIC_IDS2017.parquet")

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_validation"
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, "valid")
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, "invalid")
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, "train.parquet")
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, "test.parquet")
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, "train.parquet")
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, "test.parquet")
        self.validation_report_file_path: str = os.path.join(self.data_validation_dir, "validation_report.yaml")
        self.schema_file_path: str = os.path.join("config", "schema.yaml")

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "data_transformation"
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed", "train.npy"
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed", "test.npy"
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed_object", "preprocessing.pkl"
        )
        self.label_encoder_file_path: str = os.path.join(
            self.data_transformation_dir, "transformed_object", "label_encoder.pkl"
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "model_trainer"
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, "trained_model", "model.pkl"
        )
        self.expected_accuracy: float = 0.85
        self.overfitting_underfitting_threshold: float = 0.05

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, "model_evaluation"
        )
        self.report_file_path: str = os.path.join(self.model_evaluation_dir, "report.yaml")
        self.confusion_matrix_path: str = os.path.join(self.model_evaluation_dir, "confusion_matrix.png")

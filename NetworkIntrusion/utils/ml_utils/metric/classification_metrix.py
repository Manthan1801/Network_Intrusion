import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from NetworkIntrusion.entity.artifact_entity import ClassificationMetricArtifact
from NetworkIntrusion.exception.exception import NetworkIntrusionException

def get_classification_metric(y_true, y_pred) -> ClassificationMetricArtifact:
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

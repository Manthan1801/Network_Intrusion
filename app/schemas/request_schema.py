import os
import yaml
# pyrefly: ignore [missing-import]
from pydantic import create_model, Field
from typing import Any

SCHEMA_FILE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "schema.yaml")

def load_schema() -> dict:
    if not os.path.exists(SCHEMA_FILE_PATH):
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_FILE_PATH}")
    with open(SCHEMA_FILE_PATH, "r") as file:
        return yaml.safe_load(file)

def generate_request_model():
    """
    Dynamically generates a Pydantic model for request validation
    based entirely on the schema.yaml file generated during training.
    This completely avoids hardcoding feature names in the API.
    """
    schema = load_schema()
    columns = schema.get("columns", {})
    target_columns = schema.get("target_column", ["Label"])
    
    # Exclude the target column(s) from the request schema
    feature_columns = {k: v for k, v in columns.items() if k not in target_columns}
    
    # Map numpy/pandas types to python built-in types
    type_mapping = {
        "int64": int,
        "float64": float,
        "object": str,
        "bool": bool
    }
    
    fields = {}
    for col_name, dtype_str in feature_columns.items():
        # Clean column names for python variables if necessary (e.g. spaces to underscores)
        # But for FastAPI payload, we can use aliases
        python_type = type_mapping.get(dtype_str, float)
        fields[col_name] = (python_type, Field(..., alias=col_name))
        
    return create_model("NetworkFlowData", **fields)

# The dynamically created Pydantic Model class
NetworkFlowData = generate_request_model()

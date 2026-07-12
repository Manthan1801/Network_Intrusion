import pandas as pd
import numpy as np
import logging

def validate_features(df: pd.DataFrame, expected_columns: list) -> pd.DataFrame:
    """
    Validates feature count, order, missing values, and infinite values before inference.
    Logs all failures without crashing. Fills missing with NaN to be handled by preprocessor.
    """
    try:
        # Check for missing expected columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            logging.warning(f"Missing {len(missing_cols)} columns. Filling with NaN.")
            for col in missing_cols:
                df[col] = np.nan
        
        # Ensure exact column order
        df = df[expected_columns]
        
        # Check and log infinite values
        inf_mask = np.isinf(df.select_dtypes(include=[np.number]))
        if inf_mask.any().any():
            logging.warning("Infinite values detected in input features. Replacing with NaN.")
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
        return df
        
    except Exception as e:
        logging.error(f"Validation layer exception: {str(e)}")
        # In case of absolute failure, return the raw dataframe to let the preprocessor crash safely
        return df

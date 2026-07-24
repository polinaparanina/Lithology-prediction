import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
   
        
class DataCleansing(BaseEstimator, TransformerMixin):
    def __init__(
        self, 
        cols_to_drop: list[str] | None = None, 
        cols_to_log: list[str] | None = None
    ):
        self.cols_to_drop = cols_to_drop or []
        self.cols_to_log = cols_to_log or []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        
        for col in self.cols_to_log:
            if col in df.columns:
                df[col] = np.log1p(np.maximum(0, df[col]))
                
        existing_cols_to_drop = [col for col in self.cols_to_drop if col in df.columns]
        df = df.drop(columns=existing_cols_to_drop)

        return df
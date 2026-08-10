import pandas as pd
import numpy as np
from typing import List, Optional
from hyperimpute.plugins.imputers import Imputers

class HyperImputeWrapper:
    def __init__(self, feature_cols, method='hyperimpute', optimizer='hyperband', random_state=42):
        self.feature_cols = feature_cols
        self.method = method
        self.optimizer = optimizer
        self.random_state = random_state
        self.imputer = None
        
    def fit(self, df):
        X = df[self.feature_cols].copy()
        
        if self.method == 'hyperimpute':
            self.imputer = Imputers().get('hyperimpute', 
                                          optimizer=self.optimizer, 
                                          classifier_seed=["catboost", "random_forest", "logistic_regression"],
                                          regression_seed=["catboost", "random_forest", "linear_regression"],
                                          )
        else:
            # 'missforest', 'gain', 'mice', 'ice' и т.д.
            self.imputer = Imputers().get(self.method)
        print(f"Обучение HyperImpute ({self.method}) на {len(self.feature_cols)} признаках >>> ")
        self.imputer.fit(X)
        return self
    
    def transform(self, df):
        df_out = df.copy()
        X = df_out[self.feature_cols].copy()
        
        X_imputed = self.imputer.transform(X)
        
        if isinstance(X_imputed, pd.DataFrame):
            df_out[self.feature_cols] = X_imputed.values
        else:
            df_out[self.feature_cols] = X_imputed
            
        return df_out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
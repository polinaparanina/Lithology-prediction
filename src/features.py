import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class GeophysicalFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self 


    def transform(self, X: pd.DataFrame):
        df = X.copy()    
        
        # 1. Отношение скоростей продольной и поперечной волн Vp/Vs
        if "DTC" in df.columns and "DTS" in df.columns:
            df["VP_VS_RATIO"] = df["DTS"] / df["DTC"]
            
            vp_vs_sq = df["VP_VS_RATIO"] ** 2
            df["POISSON_RATIO"] = (0.5 * vp_vs_sq - 1) / (vp_vs_sq - 1)
            
        # 2. Акустический импеданс
        if "RHOB" in df.columns and "DTS" in df.columns:
            df["S_IMPEDANCE"] = 1e6 * df["RHOB"] / df["DTS"]

        if "RHOB" in df.columns and "DTC" in df.columns:
            df["P_IMPEDANCE"] = 1e6 * df["RHOB"] / df["DTC"]
            
        # 3. Объемный модуль упругости
        if "RHOB" in df.columns and "DTC" in df.columns and "DTS" in df.columns:
            df["BULK_MODULUS"] = (
                1e6 * df["RHOB"] * (1 / (df["DTC"] ** 2) - (4 / 3) / (df["DTS"] ** 2))
            )
            
        # 4. 
        if "RHOB" in df.columns:
            df["TOC"] = 154.497 / df["RHOB"] - 57.261
            
        # 5. Пористость по плотности
        if "NPHI" in df.columns and "RHOB" in df.columns:
            phi_d = (2.65 - df["RHOB"]) / (2.65 - 1.0)
            df["NPHI_RHOB_DIFF"] = df["NPHI"] - phi_d
            
        # 6. Прокси объема глинистости    
        if "GR" in df.columns:
            gr_min = df["GR"].quantile(0.01)
            gr_max = df["GR"].quantile(0.99)
            df["VSHALE_PROXY"] = (df["GR"] - gr_min) / (gr_max - gr_min + 1e-6)
            df["VSHALE_PROXY"] = df["VSHALE_PROXY"].clip(0, 1)
            
        return df
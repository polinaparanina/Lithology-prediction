# src/dataloader.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional


class BaseDataModule:
    """Модуль подготовки данных для CatBoost"""
    def __init__(
        self,
        target_col: Optional[str] = None,
        cat_features: Optional[List[str]] = None,
        drop_cols: Optional[List[str]] = None
    ):
        self.target_col = target_col
        self.cat_features = cat_features or []
        self.drop_cols = drop_cols or []

    def prepare_x_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        cols_to_drop = [self.target_col] + [c for c in self.drop_cols if c in df.columns]
        X = df.drop(columns=cols_to_drop)
        y = df[self.target_col] if self.target_col in df.columns else None
        return X, y

    def get_pool(self, df: pd.DataFrame):
        from catboost import Pool
        X, y = self.prepare_x_y(df)
        
        auto_cat = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
        all_cat = list(set(self.cat_features + auto_cat))
        active_cat = [c for c in all_cat if c in X.columns]

        for c in active_cat:
            X[c] = X[c].fillna("missing").astype(str)

        return Pool(data=X, label=y, cat_features=active_cat if active_cat else None)


class TensorDataset(Dataset):
    """
    PyTorch Dataset:
    - mode="autoencoder": возвращает (X, X) для реконструкции.
    - mode="lstm": нарезает 3D-тензоры [Seq_Len, Features] с учетом скважин.
    - mode="supervised": возвращает (X, y).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: Optional[str] = None,
        group_col: Optional[str] = "WELL",
        seq_len: int = 1,
        mode: str = "autoencoder"
    ):
        self.mode = mode
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.target_col = target_col

        if self.mode == "lstm" and seq_len > 1:
            self.X, self.y = self._create_sequences(df, group_col)
        else:
            self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            if target_col and target_col in df.columns:
                self.y = torch.tensor(df[target_col].values, dtype=torch.float32)
            else:
                self.y = None

    def _create_sequences(self, df: pd.DataFrame, group_col: Optional[str]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sequences_x, sequences_y = [], []
        grouped = df.groupby(group_col) if (group_col and group_col in df.columns) else [("all", df)]

        for _, group in grouped:
            features = group[self.feature_cols].values
            targets = group[self.target_col].values if (self.target_col and self.target_col in group.columns) else None

            if len(group) < self.seq_len:
                continue

            for i in range(len(group) - self.seq_len + 1):
                sequences_x.append(features[i : i + self.seq_len])
                if targets is not None:
                    sequences_y.append(targets[i + self.seq_len - 1])

        X_tensor = torch.tensor(np.array(sequences_x), dtype=torch.float32)
        y_tensor = torch.tensor(np.array(sequences_y), dtype=torch.float32) if sequences_y else None
        return X_tensor, y_tensor

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.mode == "autoencoder":
            return self.X[idx], self.X[idx]
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]
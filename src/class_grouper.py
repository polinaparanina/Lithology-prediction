import pandas as pd
import numpy as np
from typing import Dict, Union, List

class LithologyClassGrouper:
    def __init__(self, united_classes: Dict[str, Union[List, str, int]]):
        self.united_classes = united_classes
        self.class_map = self._build_reverse_map(united_classes)

    def _build_reverse_map(self, mapping: Dict) -> Dict:
        reverse_map = {}
        for group_name, items in mapping.items():
            if not isinstance(items, (list, tuple, set)):
                items = [items]
            for item in items:
                reverse_map[item] = group_name
                try:
                    num_val = float(item)
                    reverse_map[num_val] = group_name
                    reverse_map[int(num_val)] = group_name
                except (ValueError, TypeError):
                    pass
                reverse_map[str(item)] = group_name
        return reverse_map

    def transform_series(self, series: pd.Series) -> pd.Series:
        return series.map(self.class_map)

    def transform(self, df: pd.DataFrame, target_col: str = "FORCE_2020_LITHOFACIES_LITHOLOGY") -> pd.DataFrame:
        df_grouped = df.copy()
        if target_col in df_grouped.columns:
            df_grouped[target_col] = self.transform_series(df_grouped[target_col])
            df_grouped = df_grouped.dropna(subset=[target_col]).reset_index(drop=True)
        return df_grouped
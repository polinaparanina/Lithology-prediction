from os import path
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, f1_score
from src.pipelines import get_baseline_pipeline

class Baseline:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.target_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        self.group_col = 'WELL'
        self.cols_to_log = []
        self.pipeline = None
        self.is_fitted = False
        self.X_val = None
        self.y_val = None
        
        
    def fit_eval(self, df, target_col="FORCE_2020_LITHOFACIES_LITHOLOGY"):
        if self.group_col not in df.columns:
            raise ValueError(f"Колонка скважин '{self.group_col}' не найдена в датафрейме.")
        
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.test_size,
            random_state=self.random_state 
        )
        
        train_idx, val_idx = next(gss.split(df, groups=df[self.group_col]))
        
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        X_train, y_train = train_df.drop(columns=[self.target_col]), train_df[self.target_col]
        X_val, y_val = val_df.drop(columns=[self.target_col]), val_df[self.target_col]

        self.pipeline = get_baseline_pipeline(cols_to_drop=[self.group_col], cols_to_log=self.cols_to_log)
        
        self.X_val = X_val
        self.y_val = y_val
        
        print('Обучение SGDClassifier baseline >>> ')
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        preds = self.pipeline.predict(X_val)
        
        print("\n Отчет по классификации (Validation):")
        print(classification_report(y_val, preds))
        
        metrics = {
            "f1_weighted": f1_score(y_val, preds, average="weighted"),
            "f1_macro": f1_score(y_val, preds, average="macro")
        }
        return metrics
    
    
    def predict(self, df:pd.DataFrame):
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Модель ещё не обучена! Вызвать .fit_evaluate() или .load().")
        
        X = df.drop(columns=[self.target_col], errors='ignore')
        return self.pipeline.predict(X)
    
    def predict_proba(self, df: pd.DataFrame):
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Модель ещё не обучена!")

        X = df.drop(columns=[self.target_col], errors="ignore")
        return self.pipeline.predict_proba(X)
        
    
    def save(self, filepath: str|Path):
        if not self.is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Модель сохранена в: {filepath}")
        
    
    @classmethod
    def load(cls, filepath:str|Path):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
            
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        instance.is_fitted = True
        print(f"Модель загружена из: {filepath}")
        return instance
    
    
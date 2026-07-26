from pathlib import Path
import joblib
import pandas as pd
import optuna

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, f1_score
from src.pipelines import get_rf_pipeline


class RandomForest:
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.target_col = 'FORCE_2020_LITHOFACIES_LITHOLOGY'
        self.group_col = 'WELL'
        self.cols_to_log = []
        self.pipeline = None
        self.is_fitted = False
        self.X_val = None
        self.y_val = None
        
    def _split_data(self, df: pd.DataFrame):
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
        
        X_train = train_df.drop(columns=[self.target_col])
        y_train = train_df[self.target_col]
        X_val = val_df.drop(columns=[self.target_col])
        y_val = val_df[self.target_col]
        
        return X_train, y_train, X_val, y_val

    def fit_eval(self, df: pd.DataFrame, target_col: str = "FORCE_2020_LITHOFACIES_LITHOLOGY"):
        self.target_col = target_col
        X_train, y_train, X_val, y_val = self._split_data(df)

        self.pipeline = get_rf_pipeline(
            cols_to_drop=[self.group_col], 
            cols_to_log=self.cols_to_log
        )
        
        self.X_val = X_val
        self.y_val = y_val
        
        print('Обучение RandomForest (Baseline) >>> ')
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True
        
        preds = self.pipeline.predict(X_val)
        
        print("\nОтчет по классификации (Validation):")
        print(classification_report(y_val, preds))
        
        return {
            "f1_weighted": f1_score(y_val, preds, average="weighted"),
            "f1_macro": f1_score(y_val, preds, average="macro")
        }

    def tune_optuna(self, df: pd.DataFrame, n_trials: int = 10, target_col: str = "FORCE_2020_LITHOFACIES_LITHOLOGY"):
        """Подбор гиперпараметров с помощью Optuna и финальное обучение на лучших параметрах."""
        self.target_col = target_col
        X_train, y_train, X_val, y_val = self._split_data(df)
        
        self.X_val = X_val
        self.y_val = y_val

        def objective(trial):
            # Создаем временный пайплайн для каждого trial
            pipeline = get_rf_pipeline(
                cols_to_drop=[self.group_col], 
                cols_to_log=self.cols_to_log
            )

            params = {
                "model__n_estimators": trial.suggest_int("n_estimators", 100, 300, step=50),
                "model__max_depth": trial.suggest_int("max_depth", 5, 50),
                "model__min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "model__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "model__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "model__class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "model__random_state": self.random_state,
                "model__n_jobs": -1
            }

            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_val)
            
            return f1_score(y_val, preds, average="weighted")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        
        print(f"Запуск Optuna ({n_trials} итераций) >>>")
        study.optimize(objective, n_trials=n_trials)

        print("\n[Optuna] Лучшие найденные параметры:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        print(f"[Optuna] Лучший Validation F1 Weighted: {study.best_value:.4f}")

        self.pipeline = get_rf_pipeline(
            cols_to_drop=[self.group_col], 
            cols_to_log=self.cols_to_log
        )
        
        best_params_formatted = {f"model__{k}": v for k, v in study.best_params.items()}
        best_params_formatted["model__n_jobs"] = -1
        best_params_formatted["model__random_state"] = self.random_state
        
        self.pipeline.set_params(**best_params_formatted)
        
        print("\nОбучение финальной модели RandomForest >>>")
        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        preds = self.pipeline.predict(X_val)
        print("\nОтчет по классификации (Validation):")
        print(classification_report(y_val, preds))

        return {
            "best_params": study.best_params,
            "f1_weighted": f1_score(y_val, preds, average="weighted"),
            "f1_macro": f1_score(y_val, preds, average="macro")
        }

    def predict(self, df: pd.DataFrame):
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Модель ещё не обучена! Вызовите .fit_eval(), .tune_optuna() или .load().")
        
        X = df.drop(columns=[self.target_col], errors='ignore')
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame):
        if not self.is_fitted or self.pipeline is None:
            raise RuntimeError("Модель ещё не обучена!")

        X = df.drop(columns=[self.target_col], errors="ignore")
        return self.pipeline.predict_proba(X)

    def save(self, filepath: str | Path):
        if not self.is_fitted:
            raise RuntimeError("Нельзя сохранить необученную модель.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Модель сохранена в: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
            
        instance = cls()
        instance.pipeline = joblib.load(filepath)
        instance.is_fitted = True
        print(f"Модель загружена из: {filepath}")
        return instance
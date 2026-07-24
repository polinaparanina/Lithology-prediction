import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


# ----------------------------------------------------------------------
# 1. КОНФИГУРАЦИЯ КАСКАДА
# ----------------------------------------------------------------------
CASCADE_CONFIG = [
    # --- Основной каскад ---
    {"target": "DTC",  "features": ["GR", "Z_LOC"],                     "file": "catboost_final_model_dtc.cbm",  "trials": 5},
    {"target": "RHOB", "features": ["GR", "Z_LOC", "DTC"],              "file": "catboost_final_model_rhob.cbm", "trials": 5},
    {"target": "NPHI", "features": ["Z_LOC", "GR", "DTC", "RHOB"],      "file": "catboost_final_model_nphi.cbm", "trials": 5},
    {"target": "DTS",  "features": ["Z_LOC", "GR", "DTC", "RHOB", "NPHI"], "file": "catboost_final_model_dts.cbm",  "trials": 5},
    {"target": "BS",   "features": ["GR", "Z_LOC", "CALI", "DTC"],      "file": "catboost_final_model_bs.cbm",   "trials": 5},
    {"target": "CALI", "features": ["GR", "Z_LOC", "BS", "DTC"],        "file": "catboost_final_model_cali.cbm", "trials": 5},
    
    # --- Вторичный каскад (дозаполнение) ---
    {"target": "RHOB", "features": ["GR", "DEPTH_MD", "DTC"],           "file": "catboost_final_model_rhob_cascade.cbm", "trials": 3},
    {"target": "NPHI", "features": ["GR", "DEPTH_MD", "DTC"],           "file": "catboost_final_model_nphi_cascade.cbm", "trials": 3},
    {"target": "DTS",  "features": ["DEPTH_MD", "NPHI", "DTC"],         "file": "catboost_final_model_dts_cascade.cbm",  "trials": 5},
    {"target": "BS",   "features": ["DEPTH_MD", "DTC"],                 "file": "catboost_final_model_bs_cascade.cbm",   "trials": 3},
    {"target": "CALI", "features": ["DTS", "BS", "DTC"],                "file": "catboost_final_model_cali_cascade.cbm", "trials": 3},
]


def impute_column(df, target_col, feature_cols, model):
    df = df.copy()
    
    missing_mask = df[target_col].isna()
    features_available_mask = df[feature_cols].notna().all(axis=1)
    rows_to_predict = missing_mask & features_available_mask
    
    if rows_to_predict.sum() > 0:
        predictions = model.predict(df.loc[rows_to_predict, feature_cols])
        df.loc[rows_to_predict, target_col] = predictions
        
    still_missing = (df[target_col].isna()).sum()
    
    print(f"[{target_col}] Заполнено: {rows_to_predict.sum()}, осталось NaN: {still_missing}")
    
    return df


def objective(trial, X, y, groups, feature_cols, cat_features=None):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 1000),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "loss_function": "MAE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
    }
    
    gkf = GroupKFold(10)
    fold_scores = []
    
    for train_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            cat_features=cat_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=False,
        )
        
        preds = model.predict(X_val)
        fold_scores.append(mean_absolute_error(y_val, preds))
        
    return np.mean(fold_scores)



def train_and_apply_cascade(df:pd.DataFrame, models_dir:Path, config:list | None = None, cat_features: list | None = None):
    if config is None:
        config = CASCADE_CONFIG
        
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    data_imputed = df.copy()
    
    for step in config:
        target_col = step['target']
        feature_cols = step['features']
        model_filename = step['file']
        n_trials = step['trials']
        
        print(f"\n=== Обучение imputer для {target_col} по {feature_cols} ===")
        
        df_clean = data_imputed.dropna(subset=[target_col] + feature_cols + ['WELL'])
        X = df_clean[feature_cols]
        y = df_clean[target_col]
        groups = df_clean['WELL']
        
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(
            func=lambda trial: objective(trial, X, y, groups, feature_cols, cat_features), # type: ignore
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        print('=================================================================================')
        print(f"[{target_col}] MAE (CV): {study.best_value:.4f} | Параметры: {study.best_params}")
        
        
        best_params = study.best_params
        best_params.update({"loss_function": "MAE", "random_seed": 42, "verbose": False})
        
        final_model = CatBoostRegressor(**best_params)
        final_model.fit(X, y, cat_features=cat_features, verbose=False)
        
        save_path = models_dir / model_filename
        final_model.save_model(str(save_path))
        
        data_imputed = impute_column(data_imputed, target_col, feature_cols, final_model)
        
    return data_imputed


class CascadeImputer:
    def __init__(self, models_dir: Path, config: list | None = None):
        self.models_dir = Path(models_dir)
        self.config = config or CASCADE_CONFIG
        self.models = {}
        
    def load_models(self):
        for step in self.config:
            model_file = step["file"]
            model_path = self.models_dir / model_file
            if not model_path.exists():
                raise FileNotFoundError(f"Файл модели не найден: {model_path}")

            model = CatBoostRegressor()
            model.load_model(str(model_path))
            self.models[model_file] = model
    
    def transform(self, df: pd.DataFrame):
        df_imputed = df.copy()
        if not self.models:
            self.load_models()

        for step in self.config:
            target = step["target"]
            features = step["features"]
            model_file = step["file"]
            model = self.models[model_file]

            df_imputed = impute_column(df_imputed, target, features, model)

        return df_imputed
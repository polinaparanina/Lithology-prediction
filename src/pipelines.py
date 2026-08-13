# src/pipelines.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import DataCleansing 


def get_baseline_pipeline(cols_to_drop: list, cols_to_log: list) -> Pipeline:
    """
    Создает и возвращает sklearn Pipeline для бейслайн-модели SGDClassifier.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_baseline = Pipeline(
        [
            ("feature", DataCleansing(cols_to_drop=cols_to_drop, cols_to_log=cols_to_log)),
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "num",
                            numeric_transformer,
                            make_column_selector(dtype_include="number"),
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            make_column_selector(dtype_include=object),
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    max_iter=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline_baseline.named_steps["preprocessor"].set_output(transform="pandas")
    
    return pipeline_baseline



def get_rf_pipeline(cols_to_drop: list, cols_to_log: list) -> Pipeline:
    """
    Создает и возвращает sklearn Pipeline для модели RF.
    """
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),  
            ("scaler", StandardScaler()),
        ]
    )

    pipeline_rf = Pipeline(
        [
            ("feature", DataCleansing(cols_to_drop=cols_to_drop, cols_to_log=cols_to_log)),
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "num",
                            numeric_transformer,
                            make_column_selector(dtype_include="number"),
                        ),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            make_column_selector(dtype_include=object),
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
            (
            "model",
            RandomForestClassifier(random_state=42, n_jobs=-1)
            ),
        ]
    )

    pipeline_rf.named_steps["preprocessor"].set_output(transform="pandas")
    
    return pipeline_rf
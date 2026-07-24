import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

from src import (
    # train_and_apply_cascade,  # при обучении
    CascadeImputer,  # при применении
    
    GeophysicalFeatureExtractor,
    DataCleansing,
    BaseDataModule,
    TensorDataset
)

def main():
    BASE_DIR = Path(__file__).resolve().parent
    RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed" / "train_preprocessed.parquet"
    MODELS_DIR = BASE_DIR / "models" / "imputers"
    
    print(f"Базовая директория: {BASE_DIR}")
    

    print("\n[1/4] Загрузка данных...")
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {RAW_DATA_PATH}")

    raw_df = pd.read_csv(RAW_DATA_PATH, sep=';') 
    
    # print("\n[2/4] Запуск каскадной импутации (CatBoost)...")
    # imputed_df = train_and_apply_cascade(raw_df, models_dir=MODELS_DIR)
    
    print("\n[2/4] Применение каскадной импутации (загрузка готовых моделей)...")
    imputer = CascadeImputer(models_dir=MODELS_DIR)
    imputed_df = imputer.transform(raw_df)
    
    
    print("\n[3/4] Расчет геофизических признаков...")
    geo_extractor = GeophysicalFeatureExtractor()
    df_with_features = geo_extractor.transform(imputed_df)
    
    print("\n[4/4] Очистка и логарифмирование...")
    cleansing = DataCleansing(
        cols_to_drop = [
                    "X_LOC",
                    "Y_LOC",
                    "FORMATION",
                    "RDEP",
                    "SGR",
                    "RMED",
                    "ROP",
                    "PEF",
                    "RSHA",
                    "RXO",
                    "MUDWEIGHT",
                    "DCAL",
                    "RMIC",
                    "ROPA",
                    "WELL",
                    "FORCE_2020_LITHOFACIES_CONFIDENCE",
                ],
        cols_to_log = ['RDEP', 'RMED', 'RSHA']
    )
    final_df = cleansing.transform(df_with_features)
    
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"\n Пайплайн завершен! Данные сохранены в: {PROCESSED_DATA_PATH}")



    # print("\n--- Проверка загрузчиков данных (DataLoader) ---")

    # cb_module = BaseDataModule(
    #     target_col="FORCE_2020_LITHOFACIES_LITHOLOGY",
    #     drop_cols=["WELL", "DEPTH_MD"]
    # )
    # cb_pool = cb_module.get_pool(final_df)
    # print(f"CatBoost Pool готов: {cb_pool.shape[0]} строк, {len(cb_pool.get_feature_names())} признаков.")


    # feature_cols = ["GR", "RHOB", "NPHI", "DTC"] 
    # df_nn = final_df.dropna(subset=feature_cols).copy() 
    
    # lstm_dataset = TensorDataset(
    #     df=df_nn,
    #     feature_cols=feature_cols,
    #     target_col="FORCE_2020_LITHOFACIES_LITHOLOGY",
    #     group_col="WELL",
    #     seq_len=16,
    #     mode="lstm"
    # )
    # lstm_loader = DataLoader(lstm_dataset, batch_size=64, shuffle=True)

    # for X_batch, y_batch in lstm_loader:
    #     print(f"PyTorch LSTM Batch готов -> X: {X_batch.shape}, y: {y_batch.shape}")
    #     break




if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
from neuralforecast.core import NeuralForecast
from IPython.display import display
import logging
from src.get_data import get_sector_and_article_data

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def generate_all_predictions(model_save_dir: str, horizon: int):
    """
    Memuat semua model terlatih, membuat prediksi untuk setiap sektor,
    dan mengembalikan hasilnya dalam satu DataFrame.

    Args:
        data_path (str): Path ke file CSV data lengkap.
        model_save_dir (str): Path ke direktori utama tempat semua model disimpan.
        horizon (int): Jumlah hari ke depan yang akan diprediksi.

    Returns:
        pd.DataFrame: Sebuah DataFrame tunggal berisi semua prediksi, atau None jika gagal.
    """

    # --- MODEL SETTINGS ---
    settings = [
        {"sector": "Basic Materials", "feature": "GPR_Threat_Daily", "model": "NHITS"},
        {"sector": "Consumer Cyclicals", "feature": "ArticlesCount_Daily", "model": "NBEATSx"},
        {"sector": "Consumer Non-Cyclicals", "feature": "GPR_Threat_Daily", "model": "TFT"},
        {"sector": "Energy", "feature": "GPR_Threat_Daily", "model": "LSTM"},
        {"sector": "Financials", "feature": "GPR_Threat_Daily", "model": "TFT"},
        {"sector": "Industrials", "feature": "ArticlesCount_Daily", "model": "NBEATSx"},
        {"sector": "Infrastuctures", "feature": "GPR_Daily", "model": "TFT"},
        {"sector": "Kesehatan", "feature": None, "model": "LSTM"},
        {"sector": "Properties & Real Estate", "feature": "GPR_Threat_Daily", "model": "NHITS"},
        {"sector": "Technology", "feature": "GPR_Action_Daily", "model": "TFT"},
        {"sector": "Transportation & Logistic", "feature": "GPR_Action_Daily", "model": "LSTM"},
    ]

    try:
        df = get_sector_and_article_data()
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return None

    all_predictions_list = []
    print("Memulai pipeline prediksi untuk semua sektor...")

    for setting in settings:
        sector = setting['sector']
        model_type = setting['model']
        feature = setting['feature']

        model_path = os.path.join(model_save_dir, sector.replace(" & ", "_and_").replace(" ", "_"))
        print(f"\n-- Memproses {sector}... --")

        try:
            nf_loaded = NeuralForecast.load(path=model_path)
            historical_df = df[df['Sector'] == sector].copy()

            if feature:
                historical_df[feature] = historical_df[feature].rolling(window=7, min_periods=1).sum()
                historical_df.fillna(0, inplace=True)
                historical_df = historical_df.rename(columns={feature: 'x'})

            historical_df = historical_df.rename(columns={'Date': 'ds', 'Sector': 'unique_id', 'SectorVolatility_7d':'y'})
            # historical_df = historical_df.tail(40)
            predictions = nf_loaded.predict(df = historical_df)

            predictions_renamed = predictions.rename(columns={
                'ds': 'Date',
                model_type: 'SectorVolatility_7d'
            })
            predictions_renamed['Sector'] = sector
            all_predictions_list.append(predictions_renamed[['Date', 'Sector', 'SectorVolatility_7d']])
            print(f"  ✅ Prediksi untuk {sector} selesai.")
        except FileNotFoundError:
            print(f"  ⚠️ Peringatan: Model untuk '{sector}' tidak ditemukan. Melewati...")
            continue

    if not all_predictions_list:
        print("\nTidak ada prediksi yang berhasil dibuat.")
        return None,None

    final_predictions_df = pd.concat(all_predictions_list, ignore_index=True)
    final_data = df
    return final_data, final_predictions_df
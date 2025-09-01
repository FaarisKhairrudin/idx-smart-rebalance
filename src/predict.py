import pandas as pd
import numpy as np
import os
from neuralforecast.core import NeuralForecast
from IPython.display import display
import logging
from src.get_data import get_sector_and_article_data

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def generate_all_predictions(df: pd.DataFrame, model_save_dir: str, horizon: int):
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

    # try:
    #     df = get_sector_and_article_data()
    # except Exception as e:
    #     print(f"❌ ERROR: {e}")
    #     return None

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

# ================== Predict DRL ==================

import pandas as pd
import numpy as np
import joblib
from stable_baselines3 import SAC

def process_and_predict_drl(final_data, final_predictions_df, model_path, scaler_path):
    """
    Fungsi untuk memproses data menjadi df_drl dan melakukan prediksi DRL.
    
    Args:
        final_data (pd.DataFrame): Data final sektor dan artikel.
        final_predictions_df (pd.DataFrame): Data prediksi sektor.
        model_path (str): Path ke model DRL yang sudah dilatih.
        scaler_path (str): Path ke scaler yang sudah dilatih.
    
    Returns:
        pd.DataFrame: DataFrame rekomendasi proporsi alokasi dana.
    """
    # --- LANGKAH 1: Proses Data untuk df_drl ---
    df_last = final_data.sort_values('Date').groupby('Sector').tail(1)
    df_pred_last = final_predictions_df.sort_values('Date').groupby('Sector').tail(1)
    df_pred_last.rename(columns={'SectorVolatility_7d': 'vol_tplus7'}, inplace=True)

    df_long = df_last.merge(
        df_pred_last[['Sector', 'vol_tplus7']],
        on='Sector',
        how='left'
    )

    df_long.rename(columns={
        'SectorVolatility_7d': 'vol_today',
        'SectorReturn_avg': 'ret_today'
    }, inplace=True)

    settings = [
        {"sector": "Basic Materials", "feature": "GPR_Threat_Daily"},
        {"sector": "Consumer Cyclicals", "feature": "ArticlesCount_Daily"},
        {"sector": "Consumer Non-Cyclicals", "feature": "GPR_Threat_Daily"},
        {"sector": "Energy", "feature": "GPR_Threat_Daily"},
        {"sector": "Financials", "feature": "GPR_Threat_Daily"},
        {"sector": "Industrials", "feature": "ArticlesCount_Daily"},
        {"sector": "Infrastuctures", "feature": "GPR_Daily"},
        {"sector": "Kesehatan", "feature": None},
        {"sector": "Properties & Real Estate", "feature": "GPR_Threat_Daily"},
        {"sector": "Technology", "feature": "GPR_Action_Daily"},
        {"sector": "Transportation & Logistic", "feature": "GPR_Action_Daily"},
    ]

    df_long['news'] = df_long.apply(
        lambda row: row[next((s['feature'] for s in settings if s['sector'] == row['Sector']), None)] if next((s['feature'] for s in settings if s['sector'] == row['Sector']), None) is not None else 0,
        axis=1
    )

    df_to_pivot = df_long[[
        'Date', 'Sector',
        'vol_today', 'ret_today', 'news',
        'vol_tplus7',
    ]]

    df_wide = df_to_pivot.pivot(index='Date', columns='Sector')
    df_wide.columns = [f'{col[1].replace(" & ", "_").replace(" ", "_")}_{col[0]}' for col in df_wide.columns]
    df_wide.reset_index(inplace=True)
    df_drl = df_wide.dropna()
    df_drl.reset_index(drop=True, inplace=True)

    # --- LANGKAH 2: Muat Model dan Scaler ---
    print("Memuat model dan scaler...")
    model = SAC.load(model_path)
    scaler = joblib.load(scaler_path)

    SECTORS = [
        "Basic_Materials", "Consumer_Cyclicals", "Consumer_Non-Cyclicals", "Energy",
        "Financials", "Industrials", "Infrastuctures", "Kesehatan",
        "Properties_Real_Estate", "Technology", "Transportation_Logistic"
    ]

    # --- LANGKAH 3: Bentuk State ---
    print("Membentuk state untuk input model...")
    feature_cols = []
    for sector in SECTORS:
        feature_cols.append(f'{sector}_vol_today')
        feature_cols.append(f'{sector}_vol_tplus7')
        feature_cols.append(f'{sector}_ret_today')
        if sector != 'Kesehatan':
            feature_cols.append(f'{sector}_news')

    raw_market_features = df_drl[feature_cols].values.astype(np.float32)
    scaled_market_features = scaler.transform(raw_market_features.reshape(1, -1)).flatten()

    N_SECTORS = len(SECTORS)
    action_prev = np.full(N_SECTORS, 1.0 / N_SECTORS)
    final_observation = np.concatenate([scaled_market_features, action_prev]).astype(np.float32)

    # --- LANGKAH 4: Prediksi DRL ---
    print("Mendapatkan rekomendasi dari agen...")
    raw_action, _ = model.predict(final_observation, deterministic=True)
    proportions = np.exp(raw_action) / np.sum(np.exp(raw_action))

    # --- LANGKAH 5: Tampilkan Rekomendasi ---
    recommendations = pd.DataFrame({
        'Sektor': SECTORS,
        'Proporsi': proportions
    })
    recommendations['Proporsi (%)'] = recommendations['Proporsi'] * 100
    recommendations['Proporsi (%)'] = recommendations['Proporsi (%)'].map('{:,.2f}%'.format)

    return recommendations

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TFT, NHITS, NBEATSx, LSTM
from neuralforecast.losses.pytorch import MAE
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

def run_all_sector_forecast(df, settings, base_config, save_dir='./final_models', horizon=7, n_cv_windows=5):
    def smape(y_true, y_pred):
        epsilon = 1e-10
        numerator = np.abs(y_pred - y_true)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100

    def highlight_min(s):
        is_min = s == s.min()
        return ['background-color: #d4edda' if v else '' for v in is_min]

    def prepare_data(df, sector, feature):
        df_sector = df[df['Sector'] == sector].copy()
        if feature:
            df_sector[feature] = df_sector[feature].rolling(window=7, min_periods=1).sum()
            df_sector = df_sector.rename(columns={'Date': 'ds', 'SectorVolatility_7d': 'y', feature: 'x'})
            df_sector['unique_id'] = sector
            df_sector = df_sector[['unique_id', 'ds', 'y', 'x']]
        else:
            df_sector = df_sector.rename(columns={'Date': 'ds', 'SectorVolatility_7d': 'y'})
            df_sector['unique_id'] = sector
            df_sector = df_sector[['unique_id', 'ds', 'y']]
        return df_sector

    def init_model(model_type, params, scaler_type='minmax', n_blocks=[1,1,1]):
        if model_type == "TFT":
            return TFT(**params, scaler_type=scaler_type)
        elif model_type == "NHITS":
            return NHITS(**params, n_blocks=n_blocks, scaler_type=scaler_type)
        elif model_type == "NBEATSx":
            return NBEATSx(**params, n_blocks=n_blocks, scaler_type=scaler_type)
        elif model_type == "LSTM":
            return LSTM(**params, scaler_type=scaler_type)
        else:
            raise ValueError(f"Model {model_type} tidak dikenali")

    os.makedirs(save_dir, exist_ok=True)
    results = []
    fig, axes = plt.subplots(len(settings), 1, figsize=(15, 7 * len(settings)))
    if len(settings) == 1:
        axes = [axes]

    for idx, setting in enumerate(settings):
        sector, feature, model_type = setting['sector'], setting['feature'], setting['model']
        print(f"\n{'='*50}\n🔬 MEMPROSES: {sector} | MODEL: {model_type}\n{'='*50}")

        df_sector = prepare_data(df, sector, feature)

        # Model config
        common_params = {
            'h': horizon,
            'input_size': base_config['input_size'],
            'loss': MAE(),
            'max_steps': base_config['max_steps'],
            'batch_size': base_config['batch_size'],
            'random_seed': 1
        }
        if feature:
            common_params['hist_exog_list'] = ['x']
        cv_params = common_params.copy()
        cv_params['early_stop_patience_steps'] = base_config['early_stop_patience_steps']

        model_cv = init_model(model_type, cv_params, scaler_type=base_config['scaler_type'], n_blocks=base_config['n_blocks'])
        nf_cv = NeuralForecast(models=[model_cv], freq='D')

        print("  📊 Cross-validation...")
        cv_df = nf_cv.cross_validation(df=df_sector, n_windows=n_cv_windows, val_size=horizon)

        if not cv_df.empty:
            cv_df.dropna(inplace=True)
            y_true, y_pred = cv_df['y'], cv_df[model_type]
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            smape_val = smape(y_true.values, y_pred.values)

            results.append({'Sektor': sector, 'Model': model_type, 'MAE': mae, 'RMSE': rmse, 'sMAPE (%)': smape_val})
            print(f"  ✅ MAE={mae:.4f} | RMSE={rmse:.4f} | sMAPE={smape_val:.2f}%")

            # Plot window terakhir
            last_cutoff = cv_df['cutoff'].max()
            last_window_df = cv_df[cv_df['cutoff'] == last_cutoff]
            ax = axes[idx]
            last_window_df.plot(x='ds', y='y', ax=ax, label='Aktual', style='-', color='black')
            last_window_df.plot(x='ds', y=model_type, ax=ax, label='Prediksi', style='--', color='red')
            ax.set_title(f"{sector} | {model_type}\nMAE={mae:.4f} | RMSE={rmse:.4f} | sMAPE={smape_val:.2f}%")
            ax.legend()
        else:
            axes[idx].set_title(f"{sector} ({model_type}) - Gagal CV")

        print("  🚂 Training final...")
        model_final = init_model(model_type, common_params, scaler_type=base_config['scaler_type'], n_blocks=base_config['n_blocks'])
        nf_final = NeuralForecast(models=[model_final], freq='D')
        nf_final.fit(df=df_sector)
        model_path = os.path.join(save_dir, sector.replace(" & ", "_and_").replace(" ", "_"))
        nf_final.save(path=model_path, overwrite=True)
        print(f"  ✔️ Model disimpan di: {model_path}")

    plt.tight_layout(pad=3.0)
    plt.show()

    print("\n\n==================== RINGKASAN EVALUASI MODEL ====================")
    if results:
        results_df = pd.DataFrame(results).set_index(['Sektor', 'Model'])
        styled_df = results_df.style.format({
            'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'sMAPE (%)': '{:.2f}%'
        }).apply(highlight_min, subset=['MAE', 'RMSE', 'sMAPE (%)']).set_caption(
            "Perbandingan Kinerja Model (Nilai Terendah = Terbaik)"
        ).set_table_styles([{
            'selector': 'caption',
            'props': [('font-size', '16px'), ('font-weight', 'bold'), ('text-align', 'center')]
        }])
        display(styled_df)
    else:
        print("❌ Tidak ada hasil evaluasi yang valid.")


if __name__ == "__main__":
    data_path = 'data/sector_vol_with_geo_2_7d.csv'
    model_save_dir = './final_models'
    
    settings = [
        {"sector": "Basic Materials", "feature": "GPR_Threat_Daily", "model": "NHITS"},
        {"sector": "Consumer Cyclicals", "feature": "ArticlesCount_Daily", "model": "NBEATSx"},
        {"sector": "Consumer Non-Cyclicals", "feature": "GPR_Threat_Daily", "model": "TFT"},
        # {"sector": "Energy", "feature": "GPR_Threat_Daily", "model": "LSTM"},
        # {"sector": "Financials", "feature": "GPR_Threat_Daily", "model": "TFT"},
        # {"sector": "Industrials", "feature": "ArticlesCount_Daily", "model": "NBEATSx"},
        # {"sector": "Infrastuctures", "feature": "GPR_Daily", "model": "TFT"},
        # {"sector": "Kesehatan", "feature": None, "model": "LSTM"},
        # {"sector": "Properties & Real Estate", "feature": "GPR_Threat_Daily", "model": "NHITS"},
        # {"sector": "Technology", "feature": "GPR_Action_Daily", "model": "TFT"},
        # {"sector": "Transportation & Logistic", "feature": "GPR_Action_Daily", "model": "LSTM"},
    ]

    base_config = {
        'input_size': 30,
        'max_steps': 1000,
        'batch_size': 64,
        'early_stop_patience_steps': 50,
        'scaler_type': 'minmax',
        'n_blocks': [1, 1, 1]
    }

    try:
        df = pd.read_csv(data_path, parse_dates=['Date'])
    except FileNotFoundError:
        print(f"❌ ERROR: File data di '{data_path}' tidak ditemukan.")
    else:
        run_all_sector_forecast(df, settings, base_config, save_dir=model_save_dir)
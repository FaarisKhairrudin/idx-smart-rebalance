from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.predict import generate_all_predictions
import threading
import time


app = FastAPI()

# Aktifkan CORS agar frontend di port berbeda bisa akses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan ["http://127.0.0.1:8080"] untuk lebih aman
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variabel global untuk status proses
process_status = {"status": "Idle", "result": None}

def run_prediction_pipeline(model_save_dir, horizon):
    global process_status
    process_status["status"] = "Mengambil data..."
    # time.sleep(1)  # Simulasi delay, bisa dihapus
    try:
        process_status["status"] = "Melakukan prediksi..."
        final_data, final_predictions_df = generate_all_predictions(model_save_dir, horizon)
        # Konversi kolom datetime dan Timestamp ke string agar bisa di-serialize ke JSON
        import pandas as pd
        def convert_datetime(df):
            if df is not None:
                for col in df.columns:
                    if pd.api.types.is_datetime64_any_dtype(df[col]) or df[col].dtype.__class__.__name__ == 'Timestamp':
                        df[col] = df[col].astype(str)
            return df
        final_data = convert_datetime(final_data)
        final_predictions_df = convert_datetime(final_predictions_df)
        process_status["status"] = "Selesai"
        process_status["result"] = {
            "data": final_data.to_dict(orient="records") if final_data is not None else [],
            "predictions": final_predictions_df.to_dict(orient="records") if final_predictions_df is not None else []
        }
    except Exception as e:
        process_status["status"] = f"Error: {str(e)}"
        process_status["result"] = {
            "data": [],
            "predictions": []
        }

@app.get("/predict")
def predict_api():
    global process_status
    # Mulai proses prediksi di thread terpisah agar status bisa di-poll dari frontend
    if process_status["status"] in ["Idle", "Selesai", "Error"]:
        process_status = {"status": "Memulai pipeline...", "result": None}
        thread = threading.Thread(target=run_prediction_pipeline, args=("saved_models/Forecast_Model", 7))
        thread.start()
        return JSONResponse({"message": "Proses prediksi dimulai.", "status": process_status["status"]})
    else:
        return JSONResponse({"message": "Proses sedang berjalan.", "status": process_status["status"]})

@app.get("/predict/status")
def predict_status():
    global process_status
    return JSONResponse({"status": process_status["status"], "result": process_status["result"]})

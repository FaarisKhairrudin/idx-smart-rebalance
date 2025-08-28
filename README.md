# IDX PORTFOLIO REBALANCED by Tim Gacor (Telkom University)

## Cara Menjalankan Backend (FastAPI)

1. Pastikan sudah install dependensi (FastAPI, Uvicorn, dsb):
   ```zsh
   pip install -r requirements.txt
   ```
2. Jalankan backend dengan perintah:
   ```zsh
   uvicorn api_backend:app --reload --host 0.0.0.0 --port 8000
   ```
   Backend akan berjalan di http://localhost:8000

## Cara Menjalankan Frontend (Web)

1. Masuk ke folder `web`:
   ```zsh
   cd web
   ```
2. Jalankan server statis (misal dengan Python):
   ```zsh
   python3 -m http.server 8080
   ```
   Web dapat diakses di http://localhost:8080

## Akses Aplikasi
- Buka http://localhost:8080 di browser untuk frontend.
- Frontend akan otomatis terhubung ke backend di http://localhost:8000.
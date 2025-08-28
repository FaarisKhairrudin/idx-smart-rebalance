import pandas as pd
import yfinance as yf
import time
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

import os
import time
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm

def get_processed_stock_data(
    window=7,
    lookback_days=70,
    delay=0,
    sektor_csv_path="./data/Sector-Faktur.csv",
    full_data=False
):
    """
    Mengambil dan memproses data saham berdasarkan Sector-Faktur.csv
    Return: DataFrame dengan return harian + rolling sektor volatility & avg return
    """
    # --- STEP 1: Load ticker-sektor mapping
    if not os.path.exists(sektor_csv_path):
        raise FileNotFoundError(f"❌ File '{sektor_csv_path}' tidak ditemukan.")

    df_sektor = pd.read_csv(sektor_csv_path)
    ticker_to_sector = pd.Series(
        df_sektor.Sector.values,
        index=df_sektor.Faktur.str.strip()
    ).to_dict()
    tickers = [f"{ticker}.JK" for ticker in ticker_to_sector.keys()]

    # --- STEP 2: Tentukan tanggal mulai & akhir
    end_date = datetime.today().date()
    if full_data:
        # start from 2015-01-01
        start_date = datetime(2015, 1, 1).date()
    else:
        start_date = end_date - timedelta(days=lookback_days)

    # --- STEP 3: Download per ticker dengan loading bar
    successful_data = []
    
    print()
    print("🚀 Memulai pengambilan data saham...")
    print("=" * 50)
    
    with tqdm(total=len(tickers), desc="📥 Collecting data", 
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]") as pbar:
        
        for ticker_jk in tickers:
            ticker_clean = ticker_jk.replace('.JK', '')
            sector = ticker_to_sector.get(ticker_clean, 'Unknown')

            try:
                stock = yf.Ticker(ticker_jk)
                hist = stock.history(start=start_date, end=end_date)

                if hist.empty:
                    pbar.set_postfix_str(f"❌ {ticker_jk} - No data")
                    pbar.update(1)
                    continue

                df = hist.reset_index()
                df['Ticker'] = ticker_clean
                df['Sector'] = sector
                df['Date'] = pd.to_datetime(df['Date']).dt.date

                df = df[['Date', 'Ticker', 'Sector', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].round(2)
                df['Volume'] = df['Volume'].astype('int64')

                successful_data.append(df)
                pbar.set_postfix_str(f"✅ {ticker_jk} - {len(df)} records")

            except Exception as e:
                pbar.set_postfix_str(f"❌ {ticker_jk} - {e}")
            
            pbar.update(1)
            time.sleep(delay)

    
    if not successful_data:
        raise ValueError("❌ Tidak ada data saham yang berhasil diunduh.")

    # --- STEP 4: Gabungkan dan hitung return harian
    combined_df = pd.concat(successful_data, ignore_index=True)
    combined_df.sort_values(['Ticker', 'Date'], inplace=True)
    combined_df['Return'] = combined_df.groupby('Ticker')['Close'].pct_change()

    # --- STEP 5: Rolling volatility individual
    combined_df['Volatility_Individual'] = (
        combined_df.groupby('Ticker')['Return']
        .rolling(window=window)
        .std()
        .reset_index(0, drop=True)
    )

    # --- STEP 6: Hitung sektor median volatility & avg return harian
    sector_metrics = (
        combined_df.groupby(['Date', 'Sector'])[['Volatility_Individual', 'Return']]
        .median()
        .reset_index()
        .rename(columns={
            'Volatility_Individual': f'SectorVolatility_{window}d',
            'Return': 'SectorReturn_avg'
        })
    )

    # drop NaN values
    sector_metrics = sector_metrics.dropna(subset=[f'SectorVolatility_{window}d', 'SectorReturn_avg'])

    print(f"✅ Selesai! {len(successful_data)} saham berhasil diproses")
    print(f"📅 Dari: {sector_metrics['Date'].min()}")
    print(f"📅 Sampai: {sector_metrics['Date'].max()}")
    print(f"📊 Total records: {len(sector_metrics)}")
    print("=" * 50)
    print()
    return sector_metrics.reset_index(drop=True)

def download_gpr_data(lookback_days=70, full_data=False):
    """
    Mendownload dan memproses data GPR harian dari Matteo Iacoviello
    Mengembalikan DataFrame untuk N hari terakhir
    """
    import requests
    from io import BytesIO
    from tqdm import tqdm

    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"

    # print("🌐 Mengunduh file GPR dari Matteo Iacoviello...")
    
    # Loading bar untuk download
    try:
        # Baca Excel langsung dari bytes
        print("📊 Memproses data GPR...")
        print("=" * 50)
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with tqdm(total=6, desc="🔄 Processing") as pbar_proc:
            pbar_proc.update(1)

            df_raw = pd.read_excel(BytesIO(response.content))
            pbar_proc.update(1)
            
            df_raw['date'] = pd.to_datetime(df_raw['date'])
            pbar_proc.update(1)
            
            end_date = pd.to_datetime(datetime.today())
            if full_data:
                start_date = pd.to_datetime('2015-01-01')
            else:
                start_date = end_date - pd.Timedelta(days=lookback_days)
            pbar_proc.update(1)
            
            df_article = df_raw[['date', 'N10D', 'GPRD', 'GPRD_ACT', 'GPRD_THREAT']].copy()
            df_article.rename(columns={
                'date': 'Date',
                'N10D': 'ArticlesCount_Daily',
                'GPRD': 'GPR_Daily',
                'GPRD_ACT': 'GPR_Action_Daily',
                'GPRD_THREAT': 'GPR_Threat_Daily'
            }, inplace=True)
            pbar_proc.update(1)
            
            # Filter tanggal
            df_article = df_article[(df_article['Date'] >= start_date) & (df_article['Date'] <= end_date)]
            pbar_proc.update(1)

        # Proses rolling dengan progress bar
        cols_to_roll = ['ArticlesCount_Daily', 'GPR_Daily', 'GPR_Action_Daily', 'GPR_Threat_Daily']
        
        for col in cols_to_roll:
            df_article[col] = df_article[col].rolling(window=7, min_periods=1).sum()

        # Filter final
        df_filtered = df_article[
            (df_article['Date'] >= start_date) & 
            (df_article['Date'] <= end_date)
        ].copy().sort_values('Date').reset_index(drop=True)

        print(f"✅ GPR data berhasil diambil!")
        print(f"📅 Dari: {df_filtered['Date'].min()}")
        print(f"📅 Sampai: {df_filtered['Date'].max()}")
        print(f"📊 Total records: {len(df_filtered)}")
        print("=" * 50)
        
        return df_filtered

    except Exception as e:
        print("❌ Gagal mengunduh atau memproses data GPR:", e)
        return pd.DataFrame()


# buatkan fungsi untuk mendapatkan data sektor dan artikel
def get_sector_and_article_data(
    sektor_csv_path="./data/Sector-Faktur.csv",
    window=7,
    lookback_days=70,
    full_data=False,
):
    """
    Mengambil data terbaru untuk sektor dan artikel dari internet
    Args:
        sektor_csv_path (str): Path ke file CSV yang berisi data sektor.
        window (int): Jumlah hari untuk rolling sum.
        lookback_days (int): Jumlah hari ke belakang yang ingin diambil.
    """
    df_sector = get_processed_stock_data(
        sektor_csv_path=sektor_csv_path,
        window=window,
        lookback_days=lookback_days,
        full_data=full_data
    )
    df_article = download_gpr_data(lookback_days=lookback_days, full_data=full_data)

    # Pastikan kedua DataFrame tidak kosong
    if df_sector.empty or df_article.empty:
        print("❌ Gagal mengambil data sektor atau artikel.")
        return None, None

    # Pastikan kolom 'Date' ada dan bertipe datetime
    df_article['Date'] = pd.to_datetime(df_article['Date'])
    df_sector['Date'] = pd.to_datetime(df_sector['Date'])

    # join df_article dengan df_sector_vol mengunakan Date
    df_final = pd.merge(df_sector, df_article, on='Date', how='inner')
    df_final = df_final.sort_values(['Sector', 'Date']).reset_index(drop=True)

    print()
    print(f"✅ Data sektor dan artikel berhasil diperoleh: {len(df_final)} records")
    print(f"📅 Tanggal mulai: {df_final['Date'].min().date()}, Tanggal akhir: {df_final['Date'].max().date()}")
    print(f"📊 Total records per sektor: {len(df_final) / 11}")
    print("")

    return df_final



if __name__ == "__main__":
    # Contoh penggunaan (full_data=True) untuk mendapatkan data train dari 2015
    df = get_sector_and_article_data(full_data=True)

    # Contoh penggunaan (full_data=False) untuk mendapatkan data terbaru 70 hari kebelakang
    # df = get_sector_and_article_data()
    df.to_csv('./data/df_final_update.csv', index=False)

    print("✅ Data berhasil diproses dan disimpan sebagai 'processed_stock_data.csv'.")
import streamlit as st
import pandas as pd
import os
from code.predict import generate_all_predictions

# --- SETTINGS ---
SECTORS = [
    "Basic Materials",
    "Consumer Cyclicals",
    "Consumer Non-Cyclicals",
    "Energy",
    "Financials",
    "Industrials",
    "Infrastuctures",
    "Kesehatan",
    "Properties & Real Estate",
    "Technology",
    "Transportation & Logistic",
]

st.set_page_config(page_title="Rekomendasi Portofolio Saham Indonesia", layout="wide")

# Welcome & Penjelasan
st.markdown("""
<h1 style='color:#1a73e8;text-shadow:1px 1px 2px #222;'>Selamat Datang di Aplikasi Rekomendasi Portofolio Saham Indonesia</h1>
<p style='font-size:18px;color:#e0e0e0;'>
Aplikasi ini membantu Anda menganalisis dan memprediksi volatilitas sektor-sektor utama saham di Indonesia menggunakan model machine learning yang telah dilatih.
</p>
<div style='background:rgba(30,30,30,0.85);padding:18px 20px;border-radius:12px;margin-bottom:18px;border:1px solid #444;'>
<b style='color:#90caf9;font-size:17px;'>Panduan Singkat:</b>
<ul style='color:#e0e0e0;font-size:16px;'>
<li>Pilih sektor saham yang ingin dianalisis (bisa lebih dari satu).</li>
<li>Masukkan path data dan model sesuai lokasi file Anda.</li>
<li>Tekan tombol <b style='color:#90caf9;'>Lakukan Prediksi</b> untuk melihat hasil forecasting dan grafik.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Pilihan sektor di main area
sector_options = ["Semua Sektor"] + SECTORS
selected_sectors = st.multiselect(
    "*Pilih sektor yang ingin dianalisis:*", sector_options, default=["Semua Sektor"], help="Pilih satu atau beberapa sektor", label_visibility="visible"
)

st.markdown("---")
# Input path data dan model
model_save_dir = "saved_models/final_models"  # Path ke direktori model yang sudah dilatih
horizon = 7

if st.button("Lakukan Prediksi"):
    with st.spinner("Memproses prediksi..."):
        df_hist, predictions_df = generate_all_predictions(model_save_dir, horizon)
        if predictions_df is None:
            st.error("Gagal melakukan prediksi. Periksa path data dan model.")
        else:
            st.success("Prediksi berhasil!")
            # Tentukan sektor yang dipilih
            if "Semua Sektor" in selected_sectors or not selected_sectors:
                sectors_to_show = SECTORS
            else:
                sectors_to_show = selected_sectors

            filtered_df = predictions_df[predictions_df["Sector"].isin(sectors_to_show)]
            st.dataframe(filtered_df)

            # Visualisasi per sektor dalam grid 2 kolom per baris
            import altair as alt
            n_cols = 2
            cols = st.columns(n_cols)
            for idx, sector in enumerate(sectors_to_show):
                sector_df = filtered_df[filtered_df["Sector"] == sector]
                with cols[idx % n_cols]:
                    st.markdown(f"<hr style='border:1px solid #eee'>", unsafe_allow_html=True)
                    st.subheader(f"Forecasting: {sector}")
                    try:
                        # df_hist = pd.read_csv(data_path, parse_dates=['Date'])
                        df_hist_sector = df_hist[df_hist['Sector'] == sector].copy()
                        df_hist_sector = df_hist_sector.rename(columns={'SectorVolatility_7d': 'y'})
                        df_hist_sector = df_hist_sector.sort_values('Date').tail(14)
                        df_hist_sector['Tipe'] = 'Historis'
                        sector_df_plot = sector_df.copy()
                        sector_df_plot = sector_df_plot.rename(columns={'SectorVolatility_7d': 'y'})
                        sector_df_plot['Tipe'] = 'Prediksi'
                        plot_df = pd.concat([df_hist_sector[['Date','y','Tipe']], sector_df_plot[['Date','y','Tipe']]], ignore_index=True)
                        chart = alt.Chart(plot_df).mark_line().encode(
                            x='Date:T', y='y:Q', color='Tipe:N'
                        ).properties(width=350, height=250)
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Plot gagal untuk sektor {sector}: {e}")

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm # Untuk Ljung-Box, Jarque-Bera
from scipy import stats # Untuk Jarque-Bera test
from arch.univariate import ARX, NGARCH, Normal, StudentsT, SkewStudent # Menggunakan NGARCH sesuai permintaan
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model # Corrected typo: arch_modelfrom -> from arch import arch_model
from statsmodels.stats.diagnostic import het_arch # Untuk uji ARCH

# --- Konfigurasi Halaman (Hanya dipanggil sekali di awal) ---
st.set_page_config(
    page_title='Prediksi ARIMA-NGARCH Volatilitas Mata Uang 📈💰',
    page_icon='📈',
    layout="wide"
)

# --- Fungsi Pembaca Data (dengan caching) ---
@st.cache_data(ttl=86400)
def load_data(file_source, default_filename='data/default_currency_multi.csv'):
    """
    Membaca data dari objek file yang diunggah atau dari file default lokal.
    'file_source' bisa berupa uploaded_file object atau string 'default'.
    """
    df = pd.DataFrame()

    if file_source == 'default':
        path = Path(__file__).parent / default_filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.success("Data default berhasil dimuat. 🎉")
            except Exception as e:
                st.warning(f"Tidak dapat membaca file default '{default_filename}': {e} ⚠️ Pastikan formatnya benar dan tidak kosong.")
        else:
            st.warning(f"File default '{default_filename}' tidak ditemukan di {path}. Harap unggah file Anda. 📂")
    elif file_source is not None:
        try:
            df = pd.read_csv(file_source)
            st.success("File berhasil diunggah dan dibaca! ✅")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file yang diunggah: {e} ❌ Pastikan formatnya benar (CSV) dan tidak corrupt.")

    if not df.empty:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        elif df.iloc[:, 0].dtype == 'object':
            try:
                df[df.columns[0]] = pd.to_datetime(df.iloc[:, 0])
                df = df.set_index(df.columns[0])
            except Exception:
                pass
    return df

# --- Custom CSS untuk Tampilan ---
st.markdown("""
    <style>
        .css-1d3f8aq.e1fqkh3o1 {
            background-color: #f0f2f6;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .css-1v0mbdj.e1fqkh3o0 {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            border: 1px solid #d4d7dc;
            background-color: #ffffff;
            color: #333;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            text-align: left;
            margin-bottom: 0.2rem;
            transition: background-color 0.3s, color 0.3s;
        }
        .stButton>button:hover {
            background-color: #e0e6ed;
            color: #1a1a1a;
        }
        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(90, 150, 250, 0.25);
        }
        .stButton>button:active {
            background-color: #a4c6f1;
        }
        .stButton button[data-testid^="stSidebarNavButton"]:focus:not(:active) {
            background-color: #dbe9fc !important;
            font-weight: bold;
            color: #0056b3;
        }
        .main-header {
            background-color: #3f72af;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.8em;
            margin-bottom: 2rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #2c3e50;
        }
        .info-card {
            background-color: #ffffff;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            text-align: center;
            border-left: 5px solid #3f72af;
        }
        .info-card .plus-icon {
            display: none;
        }
        .interpretation-text {
            background-color: #f8f8f8;
            border-left: 5px solid #3f72af;
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        .guidance-list ul {
            list-style-type: disc;
            padding-left: 20px;
        }
        .guidance-list li {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        .guidance-list b {
            color: #3f72af;
        }
        .stTextInput>div>div>input, .stNumberInput>div>div>input {
            border-radius: 0.5rem;
            border: 1px solid #d4d7dc;
            padding: 0.75rem 1rem;
            font-size: 1rem;
        }
        .stSelectbox>div>div {
            border-radius: 0.5rem;
            border: 1px solid #d4d7dc;
            padding: 0.25rem 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Menu ---
st.sidebar.markdown("#### MENU NAVIGASI 🧭")

menu_items = {
    "HOME 🏠": "home",
    "INPUT DATA 📥": "input_data",
    "DATA PREPROCESSING 🧹": "data_preprocessing",
    "STASIONERITAS DATA 📊": "stasioneritas_data",
    "DATA SPLITTING ✂️": "data_splitting",
    "MODEL & PREDIKSI ARIMA 📈🔮": "arima_modeling_prediction", # Digabung
    "MODEL & PREDIKSI NGARCH 🌪️🔮": "ngarch_modeling_prediction", # Digabung
    "INTERPRETASI & SARAN 💡": "interpretasi_saran",
}

# Inisialisasi st.session_state jika belum ada
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

if 'selected_currency' not in st.session_state:
    st.session_state['selected_currency'] = None

if 'variable_name' not in st.session_state:
    st.session_state['variable_name'] = "Nama Variabel"

if 'df_currency_raw_multi' not in st.session_state:
    st.session_state['df_currency_raw_multi'] = pd.DataFrame()

if 'df_currency_raw' not in st.session_state:
    st.session_state['df_currency_raw'] = pd.DataFrame()

if 'cleaned_original_data' not in st.session_state: # New state for cleaned original data
    st.session_state['cleaned_original_data'] = pd.Series()

# Removed processed_returns, original_prices_for_reconstruction, return_type as they are no longer used this way
if 'processed_returns' in st.session_state:
    del st.session_state['processed_returns']
if 'original_prices_for_reconstruction' in st.session_state:
    del st.session_state['original_prices_for_reconstruction']
if 'return_type' in st.session_state:
    del st.session_state['return_type']

if 'train_data_prices' not in st.session_state: # Renamed from train_data_returns
    st.session_state['train_data_prices'] = pd.Series()

if 'test_data_prices' not in st.session_state: # Renamed from test_data_returns
    st.session_state['test_data_prices'] = pd.Series()

if 'model_arima_fit' not in st.session_state:
    st.session_state['model_arima_fit'] = None

if 'arima_residuals' not in st.session_state:
    st.session_state['arima_residuals'] = pd.Series()

if 'arima_residual_has_arch_effect' not in st.session_state:
    st.session_state['arima_residual_has_arch_effect'] = None # Untuk menyimpan hasil uji ARCH pada residual ARIMA

if 'model_ngarch_fit' not in st.session_state:
    st.session_state['model_ngarch_fit'] = None

if 'last_forecast_price_arima' not in st.session_state:
    st.session_state['last_forecast_price_arima'] = None

if 'future_predicted_prices_series' not in st.session_state:
    st.session_state['future_predicted_prices_series'] = pd.Series()

if 'predicted_prices_series' not in st.session_state:
    st.session_state['predicted_prices_series'] = pd.Series()

if 'rmse_price_arima' not in st.session_state:
    st.session_state['rmse_price_arima'] = None

if 'mae_price_arima' not in st.session_state:
    st.session_state['mae_price_arima'] = None

if 'mape_price_arima' not in st.session_state:
    st.session_state['mape_price_arima'] = None

if 'last_forecast_volatility_ngarch' not in st.session_state:
    st.session_state['last_forecast_volatility_ngarch'] = None

if 'future_predicted_volatility_series' not in st.session_state:
    st.session_state['future_predicted_volatility_series'] = pd.Series()

if 'predicted_volatility_series' not in st.session_state:
    st.session_state['predicted_volatility_series'] = pd.Series()

if 'rmse_vol_ngarch' not in st.session_state:
    st.session_state['rmse_vol_ngarch'] = None

if 'mae_vol_ngarch' not in st.session_state:
    st.session_state['mae_vol_ngarch'] = None

if 'mape_vol_ngarch' not in st.session_state:
    st.session_state['mape_vol_ngarch'] = None

for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# --- Area Konten Utama Berdasarkan Halaman yang Dipilih ---
if st.session_state['current_page'] == 'home':
    st.markdown('<div class="main-header">Prediksi Data Time Series Univariat <br> Menggunakan Model ARIMA-NGARCH 📈</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <p>Sistem ini dirancang untuk melakukan prediksi nilai tukar mata uang menggunakan model ARIMA dan mengukur volatilitasnya dengan model NGARCH. 📊💰</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">Panduan Penggunaan Sistem 💡</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="guidance-list">
    <ul>
        <li><b>HOME 🏠:</b> Halaman utama yang menjelaskan tujuan dan metode prediksi sistem.</li>
        <li><b>INPUT DATA 📥:</b> Unggah data time series nilai tukar mata uang.</li>
        <li><b>DATA PREPROCESSING 🧹:</b> Lakukan pembersihan data dari missing values dan nilai nol/negatif.</li>
        <li><b>STASIONERITAS DATA 📊:</b> Uji stasioneritas data harga awal dan periksa autokorelasi.</li>
        <li><b>DATA SPLITTING ✂️:</b> Pisahkan data harga awal menjadi latih dan uji.</li>
        <li><b>MODEL & PREDIKSI ARIMA 📈🔮:</b> Langkah-langkah untuk membentuk model ARIMA pada data harga awal (untuk prediksi nilai tukar), termasuk uji asumsi, koefisien, dan hasil prediksi.</li>
        <li><b>MODEL & PREDIKSI NGARCH 🌪️🔮:</b> Langkah-langkah untuk membentuk model NGARCH pada residual ARIMA (untuk prediksi volatilitas), termasuk uji asumsi, koefisien, dan hasil prediksi.</li>
        <li><b>INTERPRETASI & SARAN 💡:</b> Penjelasan hasil model dan rekomendasi.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state['current_page'] == 'input_data':
    st.markdown('<div class="main-header">Input Data 📥</div>', unsafe_allow_html=True)
    st.write("Di sinilah Anda dapat mengunggah data time series nilai tukar mata uang. Pastikan file CSV memiliki kolom-kolom mata uang. 📁")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state['variable_name'] = st.text_input("Nama Variabel:", value=st.session_state['variable_name'], key="variable_name_input")

    df_general = pd.DataFrame()

    uploaded_file_input_data_page = st.file_uploader("Pilih file CSV data nilai tukar Anda ⬆️", type="csv", key="input_data_uploader")

    if uploaded_file_input_data_page is not None:
        df_general = load_data(file_source=uploaded_file_input_data_page)
    elif 'df_currency_raw_multi' not in st.session_state or st.session_state['df_currency_raw_multi'].empty:
        st.info("Tidak ada file yang diunggah. Anda bisa mengunggah file Anda sendiri, atau kami akan mencoba memuat data contoh jika tersedia di repositori. ℹ️")
        if st.checkbox("Muat data contoh/default dari repositori? (Jika tersedia) ⚙️", key="load_default_checkbox"):
            df_general = load_data(file_source='default', default_filename='data/default_currency_multi.csv')
        else:
            st.info("Silakan unggah file CSV Anda untuk memulai. 👆")
            st.session_state['df_currency_raw_multi'] = pd.DataFrame()
            st.session_state['df_currency_raw'] = pd.DataFrame()
            st.stop()
    else:
        st.write("Data nilai tukar yang sudah dimuat sebelumnya: ✅")
        df_general = st.session_state['df_currency_raw_multi']

    if not df_general.empty:
        st.session_state['df_currency_raw_multi'] = df_general

        available_cols = [col for col in df_general.columns if pd.api.types.is_numeric_dtype(df_general[col])]
        if available_cols:
            current_idx = 0
            if st.session_state['selected_currency'] in available_cols:
                current_idx = available_cols.index(st.session_state['selected_currency'])
            st.session_state['selected_currency'] = st.selectbox("Pilih mata uang yang akan dianalisis: 🎯", available_cols, index=current_idx, key="currency_selector")
            
            if st.session_state['selected_currency']:
                st.session_state['df_currency_raw'] = df_general[[st.session_state['selected_currency']]].rename(columns={st.session_state['selected_currency']: 'Value'})
                st.info(f"Mata uang '{st.session_state['selected_currency']}' telah dipilih untuk analisis. 🔍")
                if st.session_state['variable_name'] == "Nama Variabel":
                    st.session_state['variable_name'] = st.session_state['selected_currency']

                with col2:
                    st.text_input("Jumlah Data yang Digunakan:", value=str(len(st.session_state['df_currency_raw'])), disabled=True)
                    if isinstance(st.session_state['df_currency_raw'].index, pd.DatetimeIndex):
                        start_date = st.session_state['df_currency_raw'].index.min().strftime('%Y-%m-%d')
                        end_date = st.session_state['df_currency_raw'].index.max().strftime('%Y-%m-%d')
                        st.text_input("Tanggal Awal Data:", value=start_date, disabled=True)
                        st.text_input("Tanggal Akhir Data:", value=end_date, disabled=True)
                    else:
                        st.text_input("Tanggal Awal Data:", value="N/A (Bukan tanggal)", disabled=True)
                        st.text_input("Tanggal Akhir Data:", value="N/A (Bukan tanggal)", disabled=True)
            else:
                st.warning("Tidak ada mata uang yang dipilih. Silakan pilih salah satu untuk melanjutkan. 🚫")
                st.session_state['df_currency_raw'] = pd.DataFrame()
        else:
            st.warning("Tidak ada kolom numerik yang terdeteksi dalam file Anda. Pastikan data nilai tukar adalah angka. ⚠️")
            st.session_state['df_currency_raw'] = pd.DataFrame()
    else:
        st.warning("Tidak ada data yang berhasil dimuat. Unggah file yang valid atau coba muat data contoh jika tersedia. 🚫")
        st.session_state['df_currency_raw_multi'] = pd.DataFrame()
        st.session_state['df_currency_raw'] = pd.DataFrame()
        with col2:
            st.text_input("Jumlah Data yang Digunakan:", value="0", disabled=True)
            st.text_input("Tanggal Awal Data:", value="N/A", disabled=True)
            st.text_input("Tanggal Akhir Data:", value="N/A", disabled=True)

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        st.subheader(f"Tampilan Data Terpilih: {st.session_state['selected_currency']} 📊")
        st.dataframe(st.session_state['df_currency_raw'])
        
        st.subheader(f"Visualisasi Data Nilai Tukar Mentah: {st.session_state['selected_currency']} 📈")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(x=st.session_state['df_currency_raw'].index, y=st.session_state['df_currency_raw']['Value'], mode='lines', name='Nilai Tukar', line=dict(color='#5d8aa8')))
        fig_raw.update_layout(title_text=f'Grafik Nilai Tukar Mentah {st.session_state["selected_currency"]}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw)

elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">Data Preprocessing ⚙️🧹</div>', unsafe_allow_html=True)
    st.write("Lakukan pembersihan data nilai tukar. Bagian ini hanya fokus pada penanganan nilai yang hilang dan nilai nol/negatif. ✨")

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        df_raw = st.session_state['df_currency_raw'].copy()
        st.write(f"Data nilai tukar mentah untuk {st.session_state.get('selected_currency', '')}: 📊")
        st.dataframe(df_raw.head())

        st.subheader("Pilih Kolom Data dan Penanganan 🔄")

        series_data = df_raw['Value']

        st.markdown("##### Penanganan Missing Values 🚫❓")
        if series_data.isnull().any():
            st.warning(f"Terdapat nilai hilang ({series_data.isnull().sum()} nilai). ⚠️ Mohon tangani:")
            missing_strategy = st.selectbox("Pilih strategi penanganan missing values:",
                                            ["Drop NA", "Isi dengan Mean", "Isi dengan Median", "Isi dengan Nilai Sebelumnya (FFill)", "Isi dengan Nilai Berikutnya (BFill)"],
                                            key="missing_strategy")
            if missing_strategy == "Drop NA":
                series_data = series_data.dropna()
                st.info("Nilai hilang dihapus. ✅")
            elif missing_strategy == "Isi dengan Mean":
                series_data = series_data.fillna(series_data.mean())
                st.info("Nilai hilang diisi dengan mean. ✅")
            elif missing_strategy == "Isi dengan Median":
                series_data = series_data.fillna(series_data.median())
                st.info("Nilai hilang diisi dengan median. ✅")
            elif missing_strategy == "Isi dengan Nilai Sebelumnya (FFill)":
                series_data = series_data.fillna(method='ffill')
                st.info("Nilai hilang diisi dengan nilai sebelumnya (forward fill). ✅")
            elif missing_strategy == "Isi dengan Nilai Berikutnya (BFill)":
                series_data = series_data.fillna(method='bfill')
                st.info("Nilai hilang diisi dengan nilai berikutnya (backward fill). ✅")
            else:
                st.info("Nilai hilang dibiarkan. 🤷")
        else:
            st.info("Tidak ada nilai hilang terdeteksi. 👍 Dataset Anda bersih!")

        st.markdown("##### Penanganan Nilai Nol atau Negatif 🚨")
        zero_or_negative_values = series_data[series_data <= 0]
        if not zero_or_negative_values.empty:
            st.warning(f"Terdapat {len(zero_or_negative_values)} nilai nol atau negatif dalam data Anda. Ini dapat menyebabkan masalah di langkah selanjutnya. ❗")
            clean_strategy = st.selectbox("Pilih strategi penanganan nilai nol/negatif:",
                                          ["Hapus baris tersebut", "Ganti dengan nilai yang sangat kecil positif (mis. 1e-6)"],
                                          key="clean_strategy")
            if clean_strategy == "Hapus baris tersebut":
                series_data = series_data[series_data > 0]
                st.info("Baris dengan nilai nol atau negatif telah dihapus. ✅")
            elif clean_strategy == "Ganti dengan nilai yang sangat kecil positif (mis. 1e-6)":
                series_data = series_data.replace(0, 1e-6)
                series_data = series_data.apply(lambda x: 1e-6 if x < 1e-6 else x)
                st.info("Nilai nol atau negatif telah diganti dengan 1e-6. ✅")
        else:
            st.info("Tidak ada nilai nol atau negatif terdeteksi. 👍 Data bersih!")

        # Store the cleaned original data in session_state
        if st.button("Selesaikan Preprocessing ▶️", key="finish_preprocessing_button"):
            st.session_state['cleaned_original_data'] = series_data
            st.success("Data harga mentah telah dibersihkan! 🎉 Siap untuk uji stasioneritas.")
            st.write("5 baris pertama data harga yang telah dibersihkan:")
            st.dataframe(series_data.head())

            st.subheader(f"Visualisasi Data Harga yang Telah Dibersihkan: {st.session_state['selected_currency']} 📈")
            fig_cleaned = go.Figure()
            fig_cleaned.add_trace(go.Scatter(x=series_data.index, y=series_data, mode='lines', name='Harga Dibersihkan', line=dict(color='#5d8aa8')))
            fig_cleaned.update_layout(title_text=f'Grafik Harga {st.session_state["selected_currency"]} Setelah Preprocessing', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_cleaned)
    else:
        st.info("Unggah data nilai tukar terlebih dahulu di bagian 'Input Data' dan pilih mata uang untuk melakukan preprocessing. ⬆️")

elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">Stasioneritas Data Harga Awal 📊🧪</div>', unsafe_allow_html=True)
    st.write(f"Untuk pemodelan time series, data harus stasioner atau dibuat stasioner melalui differencing. Kita akan menguji stasioneritas pada data harga awal {st.session_state.get('selected_currency', '')} dan memeriksa autokorelasi. 🔍")

    if 'cleaned_original_data' in st.session_state and not st.session_state['cleaned_original_data'].empty:
        series_to_test = st.session_state['cleaned_original_data']
        st.write(f"5 baris pertama data harga {st.session_state.get('selected_currency', '')} yang akan diuji:")
        st.dataframe(series_to_test.head())

        st.subheader("Uji Augmented Dickey-Fuller (ADF) 🤔")
        if st.button("Jalankan Uji ADF ▶️", key="run_adf_test"):
            try:
                result_adf = adfuller(series_to_test)
                st.write(f"**Statistik ADF:** {result_adf[0]:.4f}")
                st.write(f"**P-value:** {result_adf[1]:.4f}")
                st.write(f"**Jumlah Lags Optimal:** {result_adf[2]}")
                st.write("**Nilai Kritis:**")
                for key, value in result_adf[4].items():
                    st.write(f"  {key}: {value:.4f}")

                if result_adf[1] <= 0.05:
                    st.success("Data harga **stasioner** (tolak H0: ada akar unit). ✅ Ini jarang terjadi untuk harga mentah.")
                    st.session_state['is_stationary_adf'] = True
                else:
                    st.warning("Data harga **tidak stasioner** (gagal tolak H0: ada akar unit). ⚠️")
                    st.info("Ini adalah hasil yang umum untuk data harga. Model ARIMA akan menangani ini dengan differencing (ordo 'd' > 0).")
                    st.session_state['is_stationary_adf'] = False

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan Uji ADF: {e} ❌ Pastikan data numerik dan tidak memiliki nilai tak terbatas/NaN.")

        st.subheader("Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF) 📈📉")
        st.info("Plot ACF menunjukkan korelasi antara observasi dan observasi sebelumnya pada berbagai lag. Plot PACF menunjukkan korelasi parsial, setelah menghilangkan pengaruh korelasi dari lag yang lebih pendek. Ini membantu dalam menentukan ordo p dan q untuk model ARIMA, serta ordo differencing 'd' jika data tidak stasioner (seringkali ACF meluruh lambat dan PACF *cut off* di lag 1 untuk d=1).")
        
        lags = st.slider("Jumlah Lags untuk Plot ACF/PACF:", 5, 50, 20, key="acf_pacf_lags")
        
        if st.button("Tampilkan Plot ACF dan PACF 📊", key="show_acf_pacf"):
            try:
                fig_acf = plot_acf(series_to_test, lags=lags, alpha=0.05)
                plt.title(f'ACF {st.session_state.get("selected_currency", "")} Harga Awal')
                st.pyplot(fig_acf)

                fig_pacf = plot_pacf(series_to_test, lags=lags, alpha=0.05)
                plt.title(f'PACF {st.session_state.get("selected_currency", "")} Harga Awal')
                st.pyplot(fig_pacf)

                st.success("Plot ACF dan PACF berhasil ditampilkan! 🎉")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat plot ACF/PACF: {e} ❌ Pastikan data tidak kosong.")
    else:
        st.info("Silakan unggah dan bersihkan data terlebih dahulu di halaman 'Input Data' dan 'Data Preprocessing'. ⬆️")

elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">Data Splitting ✂️📊</div>', unsafe_allow_html=True)
    st.write(f"Pisahkan data harga awal {st.session_state.get('selected_currency', '')} menjadi set pelatihan dan pengujian untuk melatih dan mengevaluasi model ARIMA. Pembagian akan dilakukan secara berurutan karena ini adalah data time series. 📏")

    if 'cleaned_original_data' in st.session_state and not st.session_state['cleaned_original_data'].empty:
        data_to_split = st.session_state['cleaned_original_data']
        st.write(f"Data harga awal {st.session_state.get('selected_currency', '')} yang akan dibagi:")
        st.dataframe(data_to_split.head())

        st.subheader("Konfigurasi Pembagian Data ⚙️")
        test_size_ratio = st.slider("Pilih rasio data pengujian (%):", 10, 50, 20, 5, key="test_size_slider")
        test_size_frac = test_size_ratio / 100.0
        st.write(f"Rasio pengujian: {test_size_ratio}% (Data pelatihan: {100 - test_size_ratio}%)")

        if st.button("Lakukan Pembagian Data ▶️", key="split_data_button"):
            train_size = int(len(data_to_split) * (1 - test_size_frac))
            train_data_prices = data_to_split.iloc[:train_size]
            test_data_prices = data_to_split.iloc[train_size:]

            st.session_state['train_data_prices'] = train_data_prices
            st.session_state['test_data_prices'] = test_data_prices

            st.success("Data harga berhasil dibagi! ✅")
            st.write(f"Ukuran data pelatihan: {len(train_data_prices)} sampel 💪")
            st.write(f"Ukuran data pengujian: {len(test_data_prices)} sampel 🧪")

            st.subheader(f"Visualisasi Pembagian Data Harga {st.session_state.get('selected_currency', '')} Time Series 📈📉")
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(x=train_data_prices.index, y=train_data_prices.values, mode='lines', name='Data Pelatihan', line=dict(color='#3f72af')))
            fig_split.add_trace(go.Scatter(x=test_data_prices.index, y=test_data_prices.values, mode='lines', name='Data Pengujian', line=dict(color='#ff7f0e')))
            fig_split.update_layout(title_text=f'Pembagian Data Harga {st.session_state.get("selected_currency", "")} Time Series', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_split)
    else:
        st.warning("Tidak ada data harga yang tersedia untuk dibagi. Pastikan Anda telah melalui 'Input Data' dan 'Data Preprocessing'. ⚠️⬆️")

elif st.session_state['current_page'] == 'arima_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI ARIMA 📈🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model ARIMA pada data harga awal {st.session_state.get('selected_currency', '')} untuk memodelkan mean (prediksi nilai tukar), lalu lakukan prediksi. 📊")

    # Pastikan variabel-variabel ini tersedia di session_state dari halaman sebelumnya
    train_data_prices = st.session_state.get('train_data_prices', pd.Series())
    test_data_prices = st.session_state.get('test_data_prices', pd.Series())

    if not train_data_prices.empty and not test_data_prices.empty:
        # --- MODEL ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>A. Pemodelan ARIMA (Mean Equation) ⚙️</h3>", unsafe_allow_html=True)
        st.write(f"Data pelatihan harga awal untuk pemodelan ARIMA ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_prices.head())

        st.subheader("A.1. Pilih Ordo ARIMA (p, d, q) 🔢")
        st.info("Pilih kombinasi ordo ARIMA yang telah ditentukan. Ordo differencing (d) akan digunakan untuk membuat data stasioner jika diperlukan.")

        arima_orders_options = {
            "ARIMA (1,1,1)": (1, 1, 1),
            "ARIMA (0,1,1)": (0, 1, 1),
            "ARIMA (1,1,0)": (1, 1, 0),
            "ARIMA (2,1,0)": (2, 1, 0),
            "ARIMA (0,1,2)": (0, 1, 2),
            "ARIMA (2,1,1)": (2, 1, 1),
            "ARIMA (1,1,2)": (1, 1, 2),
            "ARIMA (2,1,2)": (2, 1, 2)
        }
        
        selected_arima_label = st.selectbox(
            "Pilih salah satu model ARIMA:",
            list(arima_orders_options.keys()),
            key="arima_model_selector"
        )
        
        p, d, q = arima_orders_options[selected_arima_label]
        
        st.write(f"Ordo ARIMA yang dipilih: **p={p}, d={d}, q={q}**")

        if st.button("A.2. Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_prices, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.session_state['arima_residuals'] = pd.Series(model_arima_fit.resid, index=train_data_prices.index)
                    st.success("Model ARIMA berhasil dilatih! 🎉")
                    
                    st.subheader("A.3. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    model_summary = model_arima_fit.summary()
                    # Ensure tables exist
                    if hasattr(model_summary, 'tables') and len(model_summary.tables) > 1:
                        params_table = model_summary.tables[1]
                        st.dataframe(params_table)
                    else:
                        st.info("Tidak dapat menampilkan tabel koefisien secara rinci. Silakan lihat ringkasan model di atas.")

                    # --- PREDIKSI HARGA DENGAN ARIMA ---
                    st.markdown("<h3 class='section-header'>B. Prediksi Harga dengan ARIMA 📈🔮</h3>", unsafe_allow_html=True)
                    st.info("Prediksi harga akan dilakukan pada data pengujian. Karena model ARIMA dilatih pada data harga awal dengan differencing (d>0), prediksi secara otomatis akan direkonstruksi ke skala harga asli.")

                    # Fitted values on training data
                    fitted_values_arima = model_arima_fit.predict(start=0, end=len(train_data_prices)-1, typ='levels')
                    fitted_values_arima.name = 'Fitted Values'

                    # Forecast on test data
                    forecast_steps = len(test_data_prices)
                    forecast_arima = model_arima_fit.forecast(steps=forecast_steps, typ='levels')
                    forecast_arima.index = test_data_prices.index # Assign test data index to forecast

                    st.session_state['last_forecast_price_arima'] = forecast_arima.iloc[-1]
                    st.session_state['future_predicted_prices_series'] = forecast_arima # This is the out-of-sample forecast

                    # Combine fitted values and forecast for plotting
                    combined_predictions_index = train_data_prices.index.append(test_data_prices.index)
                    predicted_prices_series = pd.concat([fitted_values_arima, forecast_arima])
                    predicted_prices_series = predicted_prices_series.reindex(combined_predictions_index) # Ensure correct indexing

                    st.session_state['predicted_prices_series'] = predicted_prices_series

                    # Visualisasi Prediksi Harga
                    st.subheader("B.1. Grafik Harga Aktual vs. Prediksi ARIMA 📊")
                    fig_price_pred = go.Figure()
                    fig_price_pred.add_trace(go.Scatter(x=train_data_prices.index, y=train_data_prices, mode='lines', name='Harga Latih Aktual', line=dict(color='blue')))
                    fig_price_pred.add_trace(go.Scatter(x=test_data_prices.index, y=test_data_prices, mode='lines', name='Harga Uji Aktual', line=dict(color='green')))
                    fig_price_pred.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series, mode='lines', name='Prediksi ARIMA', line=dict(color='red', dash='dash')))
                    fig_price_pred.update_layout(title_text=f'Harga Aktual vs. Prediksi ARIMA untuk {st.session_state["selected_currency"]}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_price_pred)

                    # Evaluasi Model (RMSE, MAE, MAPE)
                    st.subheader("B.2. Evaluasi Kinerja Prediksi Harga 📈")
                    
                    actual_test_prices = test_data_prices
                    predicted_test_prices = forecast_arima.reindex(actual_test_prices.index)
                    
                    valid_indices = actual_test_prices.index.intersection(predicted_test_prices.index).dropna()
                    actual_test_prices_filtered = actual_test_prices.loc[valid_indices]
                    predicted_test_prices_filtered = predicted_test_prices.loc[valid_indices]

                    if not actual_test_prices_filtered.empty and not predicted_test_prices_filtered.empty:
                        rmse_price = np.sqrt(np.mean((predicted_test_prices_filtered - actual_test_prices_filtered)**2))
                        mae_price = np.mean(np.abs(predicted_test_prices_filtered - actual_test_prices_filtered))
                        mape_price = np.mean(np.abs((actual_test_prices_filtered - predicted_test_prices_filtered) / actual_test_prices_filtered)) * 100

                        st.metric(label="RMSE (Harga)", value=f"{rmse_price:.4f}")
                        st.metric(label="MAE (Harga)", value=f"{mae_price:.4f}")
                        st.metric(label="MAPE (Harga)", value=f"{mape_price:.2f}%")

                        st.session_state['rmse_price_arima'] = rmse_price
                        st.session_state['mae_price_arima'] = mae_price
                        st.session_state['mape_price_arima'] = mape_price
                    else:
                        st.warning("Tidak cukup data untuk menghitung metrik evaluasi harga. Pastikan ada data pengujian yang valid.")

                    # --- UJI ASUMSI RESIDUAL ARIMA ---
                    st.markdown("<h3 class='section-header'>C. Uji Asumsi Residual ARIMA 🧪📊</h3>", unsafe_allow_html=True)
                    arima_residuals = st.session_state['arima_residuals']

                    if not arima_residuals.empty:
                        st.subheader("C.1. Visualisasi Residual ARIMA 📉")
                        fig_res_hist = go.Figure(data=[go.Histogram(x=arima_residuals, nbinsx=50, marker_color='#3f72af')])
                        fig_res_hist.update_layout(title_text='Histogram Residual ARIMA', xaxis_title='Residual', yaxis_title='Frekuensi')
                        st.plotly_chart(fig_res_hist)

                        fig_res_line = go.Figure(data=[go.Scatter(x=arima_residuals.index, y=arima_residuals, mode='lines', line=dict(color='#ff7f0e'))])
                        fig_res_line.update_layout(title_text='Plot Residual ARIMA Terhadap Waktu', xaxis_title='Tanggal', yaxis_title='Residual')
                        st.plotly_chart(fig_res_line)

                        # Uji White Noise (Ljung-Box Test)
                        st.subheader("C.2. Uji White Noise (Ljung-Box Test) 👻")
                        
                        lb_test = sm.stats.acorr_ljungbox(arima_residuals, lags=[10], return_df=True)
                        p_value_ljungbox = lb_test.iloc[0]['lb_pvalue']
                        st.write(f"**P-value Ljung-Box (lag 10):** {p_value_ljungbox:.4f}")

                        if p_value_ljungbox > 0.05:
                            st.success("Residual ARIMA adalah **white noise** (gagal tolak H0). Ini adalah hasil yang baik! ✅")
                        else:
                            st.warning("Residual ARIMA **bukan white noise** (tolak H0). ⚠️ Mungkin ada informasi yang belum dimodelkan. Pertimbangkan ordo ARIMA lain.")

                        # Uji Normalitas (Jarque-Bera Test)
                        st.subheader("C.3. Uji Normalitas Residual (Jarque-Bera Test) 🔔")
                        jb_test = stats.jarque_bera(arima_residuals)
                        st.write(f"**Jarque-Bera Statistic:** {jb_test[0]:.4f}")
                        st.write(f"**P-value:** {jb_test[1]:.4f}")
                        st.write(f"**Skewness:** {jb_test[2]:.4f}")
                        st.write(f"**Kurtosis:** {jb_test[3]:.4f}")

                        if jb_test[1] > 0.05:
                            st.success("Residual ARIMA **terdistribusi normal** (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual ARIMA **tidak terdistribusi normal** (tolak H0). ⚠️ Ini sering terjadi pada data finansial dan mengindikasikan volatilitas non-konstan atau *fat tails*, yang perlu dimodelkan dengan GARCH.")
                        
                        # Uji Heteroskedastisitas (ARCH Test)
                        st.subheader("C.4. Uji Heteroskedastisitas (ARCH Test) 🌪️")
                        st.info("Uji ARCH digunakan untuk mendeteksi apakah ada efek ARCH (Autoregressive Conditional Heteroskedasticity) pada residual. Jika ada, ini berarti volatilitasnya tidak konstan dan perlu dimodelkan dengan GARCH/NGARCH.")

                        arch_test_lags = st.slider("Jumlah Lags untuk Uji ARCH:", 1, 10, 5, key="arch_test_lags")
                        
                        if st.button("Jalankan Uji ARCH ▶️", key="run_arch_test"):
                            try:
                                # het_arch returns (lm_statistic, p_value, f_statistic, f_p_value)
                                arch_test_result = het_arch(arima_residuals.dropna(), nlags=arch_test_lags)
                                p_value_arch = arch_test_result[1]
                                st.write(f"**LM Statistic:** {arch_test_result[0]:.4f}")
                                st.write(f"**P-value (LM Test):** {p_value_arch:.4f}")

                                if p_value_arch <= 0.05:
                                    st.warning("Terdapat efek ARCH (heteroskedastisitas) pada residual ARIMA (tolak H0: homoskedastisitas). ❗ Ini mengindikasikan model NGARCH diperlukan!")
                                    st.session_state['arima_residual_has_arch_effect'] = True
                                else:
                                    st.success("Tidak ada efek ARCH (homoskedastisitas) pada residual ARIMA (gagal tolak H0). 👍")
                                    st.session_state['arima_residual_has_arch_effect'] = False
                            except Exception as e:
                                st.error(f"Terjadi kesalahan saat menjalankan Uji ARCH: {e} ❌")
                    else:
                        st.warning("Residual ARIMA tidak tersedia. Latih model ARIMA terlebih dahulu. ⚠️")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e} ❌ Pastikan data dan ordo ARIMA valid. {e}")
    else:
        st.info("Silakan unggah, bersihkan, dan bagi data terlebih dahulu di halaman 'Input Data', 'Data Preprocessing', dan 'Data Splitting'. ⬆️")

elif st.session_state['current_page'] == 'ngarch_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI NGARCH 🌪️🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual model ARIMA dari {st.session_state.get('selected_currency', '')} untuk memodelkan volatilitas (varians bersyarat) dan lakukan prediksi. 📊")

    arima_residuals = st.session_state.get('arima_residuals', pd.Series())
    test_data_prices = st.session_state.get('test_data_prices', pd.Series()) # Needed for horizon and evaluation

    if not arima_residuals.empty:
        st.markdown("<h3 class='section-header'>A. Pemodelan NGARCH (Volatilitas) ⚙️</h3>", unsafe_allow_html=True)
        st.write("Residual dari model ARIMA akan digunakan sebagai input untuk model NGARCH:")
        st.dataframe(arima_residuals.head())
        
        st.subheader("A.1. Pilih Ordo NGARCH (p, q) 🔢")
        st.info("Pilih kombinasi ordo NGARCH (p, q). P mewakili orde ARCH, Q mewakili orde GARCH.")

        ngarch_orders_options = {
            "NGARCH (1,1)": (1, 1),
            "NGARCH (1,2)": (1, 2),
            "NGARCH (2,1)": (2, 1),
            "NGARCH (2,2)": (2, 2)
        }
        
        selected_ngarch_label = st.selectbox(
            "Pilih salah satu model NGARCH:",
            list(ngarch_orders_options.keys()),
            key="ngarch_model_selector"
        )
        
        p_ngarch, q_ngarch = ngarch_orders_options[selected_ngarch_label]
        
        st.write(f"Ordo NGARCH yang dipilih: **p={p_ngarch}, q={q_ngarch}**")

        st.subheader("A.2. Pilih Distribusi Residual 📊")
        dist_options = {
            "Normal": Normal,
            "Student's T": StudentsT,
            "Skew Student's T": SkewStudent
        }
        selected_dist_label = st.selectbox(
            "Pilih distribusi residual untuk NGARCH:",
            list(dist_options.keys()),
            key="ngarch_distribution_selector"
        )
        selected_dist = dist_options[selected_dist_label]

        if st.button("A.3. Latih Model NGARCH ▶️", key="train_ngarch_button"):
            try:
                if st.session_state.get('arima_residual_has_arch_effect') == False:
                    st.warning("Uji ARCH pada residual ARIMA menunjukkan tidak ada efek ARCH. Model NGARCH mungkin tidak diperlukan atau tidak signifikan. Anda bisa tetap melatihnya, tetapi hasilnya mungkin tidak kuat.")
                
                with st.spinner("Melatih model NGARCH... ⏳"):
                    
                    model_ngarch = arch_model(
                        arima_residuals, # input is the residuals from ARIMA
                        p=p_ngarch,
                        o=1, # o=1 for NGARCH (asymmetric GARCH)
                        q=q_ngarch,
                        dist=selected_dist.__name__, # Pass the name of the distribution
                        vol='NGARCH' # Specify NGARCH for volatility model
                    )
                    model_ngarch_fit = model_ngarch.fit(disp='off') # disp='off' to suppress verbose output
                    
                    st.session_state['model_ngarch_fit'] = model_ngarch_fit
                    st.success("Model NGARCH berhasil dilatih! 🎉")

                    st.subheader("A.4. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                    st.text(model_ngarch_fit.summary().as_text())

                    st.subheader("A.5. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien NGARCH menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05.")
                    
                    model_summary_ngarch = model_ngarch_fit.summary()
                    if hasattr(model_summary_ngarch, 'tables') and len(model_summary_ngarch.tables) > 1:
                        params_table_ngarch = model_summary_ngarch.tables[1]
                        st.dataframe(params_table_ngarch)
                    else:
                        st.info("Tidak dapat menampilkan tabel koefisien NGARCH secara rinci. Silakan lihat ringkasan model di atas.")
                    
                    # --- PREDIKSI VOLATILITAS DENGAN NGARCH ---
                    st.markdown("<h3 class='section-header'>B. Prediksi Volatilitas dengan NGARCH 🌪️🔮</h3>", unsafe_allow_html=True)
                    st.info("Prediksi volatilitas akan dilakukan untuk periode data pengujian dan masa depan.")

                    # Fitted volatility on training data (implied variance)
                    fitted_volatility_ngarch = model_ngarch_fit.conditional_volatility
                    
                    # Forecast volatility for test data and future
                    if not test_data_prices.empty:
                        horizon_steps = len(test_data_prices) # For test period
                    else:
                        horizon_steps = 10 # Default future forecast if no test data for price prediction

                    forecast_ngarch_obj = model_ngarch_fit.forecast(horizon=horizon_steps) # No start date needed for out-of-sample forecast. It takes from the end of the data used for fitting.
                    forecast_volatility_ngarch = np.sqrt(forecast_ngarch_obj.variance.iloc[-1]).squeeze() # Get std dev

                    # Create index for the forecast
                    last_train_date = arima_residuals.index.max()
                    forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(days=1), periods=horizon_steps, freq='D')
                    
                    if isinstance(forecast_volatility_ngarch, np.ndarray): # Handle case where squeeze might return a scalar for horizon=1
                        forecast_volatility_ngarch = pd.Series(forecast_volatility_ngarch, index=forecast_dates)
                    elif isinstance(forecast_volatility_ngarch, pd.Series) and forecast_volatility_ngarch.index.empty:
                        forecast_volatility_ngarch.index = forecast_dates
                    elif isinstance(forecast_volatility_ngarch, pd.Series): # If it's already a series, but with default index
                        forecast_volatility_ngarch.index = forecast_dates
                    
                    st.session_state['future_predicted_volatility_series'] = forecast_volatility_ngarch
                    st.session_state['last_forecast_volatility_ngarch'] = forecast_volatility_ngarch.iloc[-1] if not forecast_volatility_ngarch.empty else None

                    # Combine fitted volatility and forecast volatility for plotting
                    combined_vol_index = fitted_volatility_ngarch.index.append(st.session_state['future_predicted_volatility_series'].index)
                    predicted_volatility_series = pd.concat([fitted_volatility_ngarch, st.session_state['future_predicted_volatility_series']])
                    predicted_volatility_series = predicted_volatility_series.reindex(combined_vol_index) # Ensure correct indexing

                    st.session_state['predicted_volatility_series'] = predicted_volatility_series

                    # Visualisasi Prediksi Volatilitas
                    st.subheader("B.1. Grafik Volatilitas Aktual (Implied) vs. Prediksi NGARCH 📊")
                    fig_vol_pred = go.Figure()
                    fig_vol_pred.add_trace(go.Scatter(x=fitted_volatility_ngarch.index, y=fitted_volatility_ngarch, mode='lines', name='Volatilitas Latih Aktual (Implied)', line=dict(color='orange')))
                    if not st.session_state['future_predicted_volatility_series'].empty:
                        fig_vol_pred.add_trace(go.Scatter(x=st.session_state['future_predicted_volatility_series'].index, y=st.session_state['future_predicted_volatility_series'], mode='lines', name='Prediksi Volatilitas NGARCH', line=dict(color='purple', dash='dash')))
                    fig_vol_pred.update_layout(title_text=f'Volatilitas Aktual vs. Prediksi NGARCH untuk {st.session_state["selected_currency"]}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_vol_pred)

                    # Evaluasi Model Volatilitas (RMSE, MAE)
                    st.subheader("B.2. Evaluasi Kinerja Prediksi Volatilitas 📈")
                    
                    # To evaluate volatility prediction, we compare forecast variance with squared residuals from the ARIMA model in the test set.
                    # First, get the residuals of the ARIMA model for the test period.
                    # We need the actual prices (test_data_prices) and ARIMA predicted prices for the test period.
                    arima_predicted_test_prices = st.session_state['predicted_prices_series'].loc[test_data_prices.index]

                    actual_test_residuals_arima = test_data_prices - arima_predicted_test_prices
                    actual_test_squared_residuals = actual_test_residuals_arima**2

                    # The NGARCH forecast for the test period is in st.session_state['future_predicted_volatility_series']
                    # We need the variance from NGARCH forecast for comparison
                    predicted_test_volatility_variance = st.session_state['future_predicted_volatility_series']**2 
                    
                    # Align indices
                    common_index = actual_test_squared_residuals.index.intersection(predicted_test_volatility_variance.index)
                    actual_test_squared_residuals_aligned = actual_test_squared_residuals.loc[common_index]
                    predicted_test_volatility_variance_aligned = predicted_test_volatility_variance.loc[common_index]

                    if not actual_test_squared_residuals_aligned.empty and not predicted_test_volatility_variance_aligned.empty:
                        rmse_vol = np.sqrt(np.mean((predicted_test_volatility_variance_aligned - actual_test_squared_residuals_aligned)**2))
                        mae_vol = np.mean(np.abs(predicted_test_volatility_variance_aligned - actual_test_squared_residuals_aligned))
                        
                        st.metric(label="RMSE (Volatilitas Variance)", value=f"{rmse_vol:.6f}")
                        st.metric(label="MAE (Volatilitas Variance)", value=f"{mae_vol:.6f}")
                        
                        st.session_state['rmse_vol_ngarch'] = rmse_vol
                        st.session_state['mae_vol_ngarch'] = mae_vol
                        st.session_state['mape_vol_ngarch'] = None # MAPE not typically used for variance evaluation
                    else:
                        st.warning("Tidak cukup data untuk menghitung metrik evaluasi volatilitas. Pastikan ada data pengujian yang valid dan prediksi ARIMA telah dilakukan.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih atau memprediksi model NGARCH: {e} ❌")
    else:
        st.info("Silakan latih model ARIMA terlebih dahulu di halaman 'Model & Prediksi ARIMA' untuk mendapatkan residual. ⬆️")

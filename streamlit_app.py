import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacff
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm # Untuk Ljung-Box, Jarque-Bera
from scipy import stats # Untuk Jarque-Bera test
from arch.univariate import ARX, NGARCH, Normal, StudentsT, SkewStudent # Menggunakan NGARCH sesuai permintaan
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
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
if 'processed_returns' not in st.session_state:
    st.session_state['processed_returns'] = pd.Series()
if 'original_prices_for_reconstruction' not in st.session_state:
    st.session_state['original_prices_for_reconstruction'] = pd.Series()
if 'return_type' not in st.session_state:
    st.session_state['return_type'] = "Log Return"
if 'train_data_returns' not in st.session_state:
    st.session_state['train_data_returns'] = pd.Series()
if 'test_data_returns' not in st.session_state:
    st.session_state['test_data_returns'] = pd.Series()
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
        <li><b>DATA PREPROCESSING 🧹:</b> Lakukan pembersihan dan transformasi data (misalnya, menghitung return).</li>
        <li><b>STASIONERITAS DATA 📊:</b> Uji stasioneritas data return dan periksa autokorelasi.</li>
        <li><b>DATA SPLITTING ✂️:</b> Pisahkan data menjadi latih dan uji.</li>
        <li><b>MODEL & PREDIKSI ARIMA 📈🔮:</b> Langkah-langkah untuk membentuk model ARIMA pada data return (untuk prediksi nilai tukar), termasuk uji asumsi, koefisien, dan hasil prediksi.</li>
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
    st.write("Lakukan pembersihan dan transformasi data nilai tukar. Untuk model ARIMA-NGARCH, kita perlu mengubah data harga menjadi return (perubahan logaritmik atau persentase). ✨")

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        df_raw = st.session_state['df_currency_raw'].copy()
        st.write(f"Data nilai tukar mentah untuk {st.session_state.get('selected_currency', '')}: 📊")
        st.dataframe(df_raw.head())

        st.subheader("Pilih Kolom Data dan Transformasi 🔄")

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
            st.warning(f"Terdapat {len(zero_or_negative_values)} nilai nol atau negatif dalam data Anda. Ini akan menyebabkan masalah saat menghitung return logaritmik atau persentase. ❗")
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
            st.info("Tidak ada nilai nol atau negatif terdeteksi. 👍 Data siap untuk transformasi!")

        st.subheader("Transformasi Data: Harga ke Return 💰➡️📊")
        return_type = st.radio("Pilih tipe return:", ("Log Return", "Simple Return"), key="return_type_radio")

        if st.button("Hitung Return ▶️", key="calculate_return_button"):
            if len(series_data) > 1:
                processed_series = pd.Series([], dtype=float)
                if return_type == "Log Return":
                    processed_series = np.log(series_data / series_data.shift(1))
                    st.info("Data telah diubah menjadi Log Return. 📈")
                else:
                    processed_series = series_data.pct_change()
                    st.info("Data telah diubah menjadi Simple Return (Persentase Perubahan). 💹")

                processed_series = processed_series.replace([np.inf, -np.inf], np.nan).dropna()

                if not processed_series.empty:
                    st.session_state['processed_returns'] = processed_series
                    st.session_state['original_prices_for_reconstruction'] = series_data
                    st.session_state['return_type'] = return_type
                    st.success("Data return berhasil dihitung! 🎉 Siap untuk analisis selanjutnya.")
                    st.write("5 baris pertama data return:")
                    st.dataframe(processed_series.head())

                    st.subheader(f"Visualisasi Data Return: {st.session_state['selected_currency']} 📉")
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Scatter(x=processed_series.index, y=processed_series, mode='lines', name='Data Return', line=dict(color='#82c0cc')))
                    fig_returns.update_layout(title_text=f'Grafik Data Return {st.session_state["selected_currency"]}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_returns)
                else:
                    st.warning("Data return kosong setelah transformasi. Pastikan data input Anda valid. ⚠️")
            else:
                st.warning("Data terlalu pendek untuk menghitung return. Minimal 2 observasi dibutuhkan. 🤏")
        else:
            st.info("Klik 'Hitung Return' untuk melanjutkan ke transformasi data. ➡️")
    else:
        st.info("Unggah data nilai tukar terlebih dahulu di bagian 'Input Data' dan pilih mata uang untuk melakukan preprocessing. ⬆️")

elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">Stasioneritas Data Return 📊🧪</div>', unsafe_allow_html=True)
    st.write(f"Untuk pemodelan time series, data harus stasioner. Kita akan menguji stasioneritas pada data return {st.session_state.get('selected_currency', '')} dan memeriksa autokorelasi. 🔍")

    if 'processed_returns' in st.session_state and not st.session_state['processed_returns'].empty:
        series_to_test = st.session_state['processed_returns']
        st.write(f"5 baris pertama data return {st.session_state.get('selected_currency', '')} yang akan diuji:")
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
                    st.success("Data return **stasioner** (tolak H0: ada akar unit). Ini adalah hasil yang baik! ✅")
                    st.session_state['is_stationary_adf'] = True
                else:
                    st.warning("Data return **tidak stasioner** (gagal tolak H0: ada akar unit). ⚠️")
                    st.info("Meskipun data return seringkali stasioner, jika tidak, Anda mungkin perlu transformasi tambahan (misalnya, differencing pada return, yang jarang terjadi).")
                    st.session_state['is_stationary_adf'] = False

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan Uji ADF: {e} ❌ Pastikan data numerik dan tidak memiliki nilai tak terbatas/NaN.")

        st.subheader("Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF) 📈📉")
        st.info("Plot ACF menunjukkan korelasi antara observasi dan observasi sebelumnya pada berbagai lag. Plot PACF menunjukkan korelasi parsial, setelah menghilangkan pengaruh korelasi dari lag yang lebih pendek. Ini membantu dalam menentukan ordo p dan q untuk model ARIMA.")
        
        lags = st.slider("Jumlah Lags untuk Plot ACF/PACF:", 5, 50, 20, key="acf_pacf_lags")
        
        if st.button("Tampilkan Plot ACF dan PACF 📊", key="show_acf_pacf"):
            try:
                fig_acf = plot_acf(series_to_test, lags=lags, alpha=0.05)
                plt.title(f'ACF {st.session_state.get("selected_currency", "")} Returns')
                st.pyplot(fig_acf)

                fig_pacf = plot_pacf(series_to_test, lags=lags, alpha=0.05)
                plt.title(f'PACF {st.session_state.get("selected_currency", "")} Returns')
                st.pyplot(fig_pacf)

                st.success("Plot ACF dan PACF berhasil ditampilkan! 🎉")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat plot ACF/PACF: {e} ❌ Pastikan data tidak kosong.")
    else:
        st.info("Silakan unggah dan proses data (hitung return) terlebih dahulu di halaman 'Input Data' dan 'Data Preprocessing'. ⬆️")

elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">Data Splitting ✂️📊</div>', unsafe_allow_html=True)
    st.write(f"Pisahkan data return {st.session_state.get('selected_currency', '')} menjadi set pelatihan dan pengujian untuk melatih dan mengevaluasi model ARIMA. Pembagian akan dilakukan secara berurutan karena ini adalah data time series. 📏")

    if 'processed_returns' in st.session_state and not st.session_state['processed_returns'].empty:
        data_to_split = st.session_state['processed_returns']
        st.write(f"Data return {st.session_state.get('selected_currency', '')} yang akan dibagi:")
        st.dataframe(data_to_split.head())

        st.subheader("Konfigurasi Pembagian Data ⚙️")
        test_size_ratio = st.slider("Pilih rasio data pengujian (%):", 10, 50, 20, 5, key="test_size_slider")
        test_size_frac = test_size_ratio / 100.0
        st.write(f"Rasio pengujian: {test_size_ratio}% (Data pelatihan: {100 - test_size_ratio}%)")

        if st.button("Lakukan Pembagian Data ▶️", key="split_data_button"):
            train_size = int(len(data_to_split) * (1 - test_size_frac))
            train_data_returns = data_to_split.iloc[:train_size]
            test_data_returns = data_to_split.iloc[train_size:]

            st.session_state['train_data_returns'] = train_data_returns
            st.session_state['test_data_returns'] = test_data_returns

            st.success("Data return berhasil dibagi! ✅")
            st.write(f"Ukuran data pelatihan: {len(train_data_returns)} sampel 💪")
            st.write(f"Ukuran data pengujian: {len(test_data_returns)} sampel 🧪")

            st.subheader(f"Visualisasi Pembagian Data Return {st.session_state.get('selected_currency', '')} 📈📉")
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(x=train_data_returns.index, y=train_data_returns.values, mode='lines', name='Data Pelatihan', line=dict(color='#3f72af')))
            fig_split.add_trace(go.Scatter(x=test_data_returns.index, y=test_data_returns.values, mode='lines', name='Data Pengujian', line=dict(color='#ff7f0e')))
            fig_split.update_layout(title_text=f'Pembagian Data Return {st.session_state.get("selected_currency", "")} Time Series', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_split)
    else:
        st.warning("Tidak ada data return yang tersedia untuk dibagi. Pastikan Anda telah melalui 'Input Data' dan 'Data Preprocessing'. ⚠️⬆️")

elif st.session_state['current_page'] == 'arima_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI ARIMA 📈🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model ARIMA pada data return {st.session_state.get('selected_currency', '')} untuk memodelkan mean (prediksi nilai tukar), lalu lakukan prediksi. 📊")

    # Pastikan variabel-variabel ini tersedia di session_state dari halaman sebelumnya
    train_data_returns = st.session_state.get('train_data_returns', pd.Series())
    test_data_returns = st.session_state.get('test_data_returns', pd.Series())
    original_prices_series = st.session_state.get('original_prices_for_reconstruction', pd.Series())
    return_type = st.session_state.get('return_type', "Log Return")

    if not train_data_returns.empty and not test_data_returns.empty and not original_prices_series.empty:
        # --- MODEL ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>A. Pemodelan ARIMA (Mean Equation) ⚙️</h3>", unsafe_allow_html=True)
        st.write(f"Data pelatihan return untuk pemodelan ARIMA ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_returns.head())

        st.subheader("A.1. Pilih Ordo ARIMA (p, d, q) 🔢")
        st.info("Pilih kombinasi ordo ARIMA yang telah ditentukan. Ordo differencing (d) akan otomatis diatur ke 0 karena Anda bekerja dengan data return yang diharapkan stasioner.")

        arima_orders_options = {
            "ARIMA (1,1,1)": (1, 0, 1),
            "ARIMA (0,1,1)": (0, 0, 1),
            "ARIMA (1,1,0)": (1, 0, 0),
            "ARIMA (2,1,0)": (2, 0, 0),
            "ARIMA (0,1,2)": (0, 0, 2),
            "ARIMA (2,1,1)": (2, 0, 1),
            "ARIMA (1,1,2)": (1, 0, 2),
            "ARIMA (2,1,2)": (2, 0, 2)
        }
        
        selected_arima_label = st.selectbox(
            "Pilih salah satu model ARIMA:",
            list(arima_orders_options.keys()),
            key="arima_model_selector"
        )
        
        # Get the p, d, q values from the selected label
        p, d, q = arima_orders_options[selected_arima_label]
        
        st.write(f"Ordo ARIMA yang dipilih: **p={p}, d={d}, q={q}**")

        if st.button("A.2. Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.session_state['arima_residuals'] = pd.Series(model_arima_fit.resid, index=train_data_returns.index)
                    st.success("Model ARIMA berhasil dilatih! 🎉")
                    
                    st.subheader("A.3. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    model_summary = model_arima_fit.summary()
                    # Ensure tables exist and get the correct table (usually the second one for parameters)
                    if len(model_summary.tables) > 1:
                        results_table_html = model_summary.tables[1].as_html()  
                        df_results = pd.read_html(results_table_html, header=0, index_col=0)[0]
                        st.dataframe(df_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    else:
                        st.warning("Tidak dapat menampilkan tabel signifikansi koefisien. Ringkasan model mungkin tidak memiliki format yang diharapkan. 🙁")

                    st.subheader("A.5. Analisis Residual ARIMA (Uji Asumsi) 🔬")
                    st.info("Residual model ARIMA harus menyerupai deret white noise (tidak berkorelasi) dan berdistribusi normal. Kehadiran autokorelasi atau heteroskedastisitas pada residual menunjukkan bahwa model ARIMA belum sepenuhnya menangkap seluruh struktur data, dan mungkin memerlukan model GARCH untuk volatilitas.")

                    residuals = st.session_state['arima_residuals']
                    if not residuals.empty:
                        # Plot Residual
                        st.write("##### Plot Residual ARIMA 📉")
                        fig_res = go.Figure()
                        fig_res.add_trace(go.Scatter(x=residuals.index, y=residuals, mode='lines', name='Residual', line=dict(color='#8856a7')))
                        fig_res.update_layout(title_text='Residual Model ARIMA', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_res)

                        col_res_1, col_res_2 = st.columns(2)

                        with col_res_1:
                            st.write("##### Uji White Noise (Ljung-Box) 🧪")
                            st.info("Menguji apakah residual tidak memiliki autokorelasi yang signifikan. P-value < 0.05 menunjukkan adanya autokorelasi, yang berarti model belum optimal.")
                            try:
                                lb_test = sm.stats.acorr_ljungbox(residuals, lags=[10], return_df=True)
                                st.write(f"**P-value Ljung-Box (lag 10):** {lb_test['lb_pvalue'].iloc[0]:.4f}")
                                if lb_test['lb_pvalue'].iloc[0] < 0.05:
                                    st.warning("Residual **tidak menyerupai white noise** (ada autokorelasi). Model ARIMA mungkin perlu disempurnakan. ⚠️")
                                else:
                                    st.success("Residual **menyerupai white noise** (tidak ada autokorelasi signifikan). 👍")
                            except Exception as e:
                                st.error(f"Gagal menjalankan Uji Ljung-Box: {e} ❌")

                        with col_res_2:
                            st.write("##### Uji Normalitas (Jarque-Bera) 🔔")
                            st.info("Menguji apakah residual berdistribusi normal. P-value < 0.05 menunjukkan non-normalitas (residual mungkin tidak normal).")
                            try:
                                jb_test = stats.jarque_bera(residuals)
                                st.write(f"**Statistik JB:** {jb_test[0]:.4f}")
                                st.write(f"**P-value JB:** {jb_test[1]:.4f}")
                                if jb_test[1] < 0.05:
                                    st.warning("Residual **tidak berdistribusi normal**. Ini umum dalam data keuangan dan menunjukkan volatilitas (perlu GARCH). ⚠️")
                                else:
                                    st.success("Residual **berdistribusi normal**. 👍")
                            except Exception as e:
                                st.error(f"Gagal menjalankan Uji Jarque-Bera: {e} ❌")
                        
                        st.write("##### Uji Heteroskedastisitas (ARCH Test) 🌪️")
                        st.info("Menguji apakah varians residual konstan (homoskedastisitas) atau bervariasi seiring waktu (heteroskedastisitas/ARCH effect). P-value < 0.05 menunjukkan adanya ARCH effect, yang berarti Anda memerlukan model GARCH.")
                        try:
                            # Use het_arch from statsmodels
                            arch_test_result = het_arch(residuals)
                            lm_statistic = arch_test_result[0]
                            p_value = arch_test_result[1]
                            f_statistic = arch_test_result[2]
                            f_p_value = arch_test_result[3]

                            st.write(f"**LM Statistic:** {lm_statistic:.4f}")
                            st.write(f"**P-value (LM Test):** {p_value:.4f}")
                            # st.write(f"F-Statistic: {f_statistic:.4f}")
                            # st.write(f"P-value (F-Test): {f_p_value:.4f}")

                            if p_value < 0.05:
                                st.error("Terdapat indikasi **heteroskedastisitas (ARCH effect)** pada residual ARIMA (P-value < 0.05). Ini berarti varians residual tidak konstan dan sangat direkomendasikan untuk melanjutkan dengan model NGARCH. 🎯")
                                st.session_state['arima_residual_has_arch_effect'] = True
                            else:
                                st.success("Tidak ada indikasi heteroskedastisitas (ARCH effect) pada residual ARIMA. Varians residual konstan. 👍")
                                st.session_state['arima_residual_has_arch_effect'] = False
                        except Exception as e:
                            st.error(f"Gagal menjalankan Uji ARCH: {e} ❌ Pastikan residual tidak kosong atau memiliki varians nol.")
                    else:
                        st.warning("Residual ARIMA tidak tersedia untuk analisis. Latih model terlebih dahulu. ⚠️")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e} ❌ Pastikan data pelatihan valid dan ordo ARIMA sesuai.")

        # --- PREDIKSI HARGA ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>B. Prediksi Harga Menggunakan ARIMA 📈</h3>", unsafe_allow_html=True)
        if st.session_state['model_arima_fit'] is not None:
            st.write("Model ARIMA telah dilatih. Sekarang kita akan melakukan prediksi pada data pengujian dan memprediksi harga masa depan.")

            predict_button_key = "predict_arima_button_trained" if st.session_state['model_arima_fit'] else "predict_arima_button_no_model"

            if st.button("B.1. Lakukan Prediksi Harga ▶️", key=predict_button_key):
                with st.spinner("Melakukan prediksi harga dengan ARIMA... ⏳"):
                    try:
                        # Prediksi pada data uji (return)
                        start_test_index = test_data_returns.index[0]
                        end_test_index = test_data_returns.index[-1]
                        
                        forecast_returns = st.session_state['model_arima_fit'].predict(start=start_test_index, end=end_test_index)
                        
                        # Pastikan forecast_returns memiliki indeks yang sama dengan test_data_returns
                        forecast_returns = pd.Series(forecast_returns, index=test_data_returns.index)

                        # Rekonstruksi harga dari return yang diprediksi
                        # Kita perlu harga terakhir dari data training untuk memulai rekonstruksi
                        last_train_price = original_prices_series.loc[train_data_returns.index[-1]]

                        predicted_prices = [last_train_price]
                        for r in forecast_returns:
                            if return_type == "Log Return":
                                predicted_prices.append(predicted_prices[-1] * np.exp(r))
                            else: # Simple Return
                                predicted_prices.append(predicted_prices[-1] * (1 + r))
                        
                        # Buang harga awal yang merupakan harga terakhir dari training set
                        predicted_prices = predicted_prices[1:]
                        predicted_prices_series = pd.Series(predicted_prices, index=test_data_returns.index)
                        st.session_state['predicted_prices_series'] = predicted_prices_series

                        st.success("Prediksi harga berhasil dilakukan! 🎉")

                        st.subheader("B.2. Evaluasi Kinerja Prediksi Harga 📊")
                        # Pastikan data aktual dan prediksi sejajar
                        actual_prices_test = original_prices_series.loc[test_data_returns.index]

                        # Hitung metrik evaluasi
                        rmse = np.sqrt(mean_squared_error(actual_prices_test, predicted_prices_series))
                        mae = np.mean(np.abs(actual_prices_test - predicted_prices_series))
                        # MAPE: Hindari pembagian nol
                        mape = np.mean(np.abs((actual_prices_test - predicted_prices_series) / actual_prices_test.replace(0, np.nan).dropna())) * 100
                        if np.isnan(mape): # Handle cases where all actual_prices_test are 0 or NaNs
                             mape = float('inf') # Or handle as appropriate for your domain

                        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.4f}")
                        st.metric("MAE (Mean Absolute Error)", f"{mae:.4f}")
                        st.metric("MAPE (Mean Absolute Percentage Error)", f"{mape:.2f}%")

                        st.session_state['rmse_price_arima'] = rmse
                        st.session_state['mae_price_arima'] = mae
                        st.session_state['mape_price_arima'] = mape
                        
                        st.subheader("B.3. Visualisasi Prediksi Harga ARIMA 📈")
                        fig_forecast_price = go.Figure()
                        fig_forecast_price.add_trace(go.Scatter(x=original_prices_series.index, y=original_prices_series, mode='lines', name='Harga Aktual (Keseluruhan)', line=dict(color='#5d8aa8')))
                        fig_forecast_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series, mode='lines', name='Harga Prediksi ARIMA', line=dict(color='red', dash='dot')))
                        fig_forecast_price.update_layout(title_text=f'Harga Aktual vs. Prediksi ARIMA ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_forecast_price)

                        # --- Prediksi Harga Masa Depan ---
                        st.subheader("B.4. Prediksi Harga Masa Depan ⏩")
                        num_future_steps = st.number_input("Jumlah langkah prediksi ke depan:", min_value=1, max_value=30, value=7, key="num_future_steps_arima")
                        
                        if st.button(f"Prediksi {num_future_steps} Hari ke Depan", key="predict_future_arima_button"):
                            with st.spinner(f"Memprediksi {num_future_steps} harga ke depan... ⏳"):
                                try:
                                    # Gunakan model_arima_fit untuk memprediksi return masa depan
                                    future_returns_forecast = st.session_state['model_arima_fit'].predict(
                                        start=len(train_data_returns),
                                        end=len(train_data_returns) + num_future_steps - 1
                                    )
                                    
                                    # Dapatkan tanggal terakhir dari data asli
                                    last_date = original_prices_series.index[-1]
                                    
                                    # Buat indeks tanggal masa depan
                                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_future_steps, freq='D')
                                    future_returns_series = pd.Series(future_returns_forecast, index=future_dates)

                                    # Rekonstruksi harga masa depan
                                    last_actual_price = original_prices_series.iloc[-1]
                                    future_predicted_prices = [last_actual_price]
                                    for r in future_returns_series:
                                        if return_type == "Log Return":
                                            future_predicted_prices.append(future_predicted_prices[-1] * np.exp(r))
                                        else: # Simple Return
                                            future_predicted_prices.append(future_predicted_prices[-1] * (1 + r))
                                    
                                    future_predicted_prices_series = pd.Series(future_predicted_prices[1:], index=future_dates)
                                    st.session_state['future_predicted_prices_series'] = future_predicted_prices_series
                                    st.session_state['last_forecast_price_arima'] = future_predicted_prices_series.iloc[-1]

                                    st.success(f"Prediksi harga untuk {num_future_steps} hari ke depan berhasil! 🎉")
                                    st.dataframe(future_predicted_prices_series)

                                    fig_future_price = go.Figure()
                                    fig_future_price.add_trace(go.Scatter(x=original_prices_series.index, y=original_prices_series, mode='lines', name='Harga Aktual (Historis)', line=dict(color='#5d8aa8')))
                                    fig_future_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series, mode='lines', name='Harga Prediksi (Data Uji)', line=dict(color='red', dash='dot')))
                                    fig_future_price.add_trace(go.Scatter(x=future_predicted_prices_series.index, y=future_predicted_prices_series, mode='lines', name='Harga Prediksi (Masa Depan)', line=dict(color='green', dash='solid')))
                                    fig_future_price.update_layout(title_text=f'Prediksi Harga {st.session_state.get("selected_currency", "")} (Historis & Masa Depan)', xaxis_rangeslider_visible=True)
                                    st.plotly_chart(fig_future_price)
                                
                                except Exception as e:
                                    st.error(f"Terjadi kesalahan saat memprediksi harga masa depan: {e} ❌")
                        else:
                            st.info("Masukkan jumlah langkah dan klik tombol untuk memprediksi harga masa depan. 🗓️")

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat melakukan prediksi harga: {e} ❌")
            else:
                st.info("Klik 'Lakukan Prediksi Harga' untuk melihat hasil dan metrik. ➡️")
        else:
            st.warning("Model ARIMA belum dilatih. Silakan latih model terlebih dahulu di bagian 'A.2. Latih Model ARIMA'. ⚠️")
    else:
        st.warning("Data pelatihan, pengujian, atau data harga asli belum tersedia. Pastikan Anda telah melalui 'Input Data', 'Data Preprocessing', dan 'Data Splitting'. ⚠️⬆️")

elif st.session_state['current_page'] == 'ngarch_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI NGARCH 🌪️🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual ARIMA dari {st.session_state.get('selected_currency', '')} untuk memodelkan volatilitas, kemudian lakukan prediksi volatilitas masa depan. 📊")

    arima_residuals = st.session_state.get('arima_residuals', pd.Series())
    arima_residual_has_arch_effect = st.session_state.get('arima_residual_has_arch_effect', None)

    if not arima_residuals.empty:
        if arima_residual_has_arch_effect is True:
            st.info("Uji ARCH pada residual ARIMA menunjukkan adanya efek heteroskedastisitas. Anda dapat melanjutkan dengan pemodelan NGARCH.")
            st.write("Residual ARIMA yang akan dimodelkan dengan NGARCH:")
            st.dataframe(arima_residuals.head())

            st.subheader("A. Pemodelan NGARCH (Variance Equation) ⚙️")
            st.info("Pilih ordo p dan q untuk model NGARCH. Model NGARCH(p,q) memodelkan varians bersyarat berdasarkan kuadrat residual masa lalu (p) dan varians bersyarat masa lalu (q).")
            
            p_ngarch = st.number_input("Ordo NGARCH (p):", min_value=1, max_value=5, value=1, key="ngarch_p")
            q_ngarch = st.number_input("Ordo NGARCH (q):", min_value=1, max_value=5, value=1, key="ngarch_q")

            if st.button("A.2. Latih Model NGARCH ▶️", key="train_ngarch_button"):
                with st.spinner("Melatih model NGARCH... ⏳"):
                    try:
                        # Model NGARCH perlu data residual
                        model_ngarch = arch_model(arima_residuals, x=None, p=p_ngarch, o=1, q=q_ngarch, dist='normal', vol='NGARCH')
                        model_ngarch_fit = model_ngarch.fit(disp='off') # disp='off' untuk menekan output verbose

                        st.session_state['model_ngarch_fit'] = model_ngarch_fit
                        st.success("Model NGARCH berhasil dilatih! 🎉")

                        st.subheader("A.3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                        st.text(model_ngarch_fit.summary().as_text())
                        
                        st.subheader("A.4. Uji Signifikansi Koefisien NGARCH (P-value) ✅❌")
                        st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                        
                        ngarch_summary_html = model_ngarch_fit.summary().tables[1].as_html()
                        df_ngarch_results = pd.read_html(ngarch_summary_html, header=0, index_col=0)[0]
                        st.dataframe(df_ngarch_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e} ❌ Pastikan ordo NGARCH sesuai dan data residual valid.")
            else:
                st.info("Pilih ordo NGARCH dan klik 'Latih Model NGARCH' untuk memulai. ➡️")
            
            # --- PREDIKSI VOLATILITAS NGARCH SECTION ---
            st.markdown("<h3 class='section-header'>B. Prediksi Volatilitas Menggunakan NGARCH 🌪️</h3>", unsafe_allow_html=True)
            if st.session_state['model_ngarch_fit'] is not None:
                st.write("Model NGARCH telah dilatih. Sekarang kita akan memprediksi volatilitas masa depan.")

                if st.button("B.1. Lakukan Prediksi Volatilitas ▶️", key="predict_volatility_button"):
                    with st.spinner("Memprediksi volatilitas dengan NGARCH... ⏳"):
                        try:
                            # Prediksi volatilitas in-sample (untuk plotting)
                            in_sample_forecast = st.session_state['model_ngarch_fit'].conditional_volatility
                            predicted_volatility_series = pd.Series(in_sample_forecast, index=arima_residuals.index)
                            st.session_state['predicted_volatility_series'] = predicted_volatility_series

                            st.success("Prediksi volatilitas in-sample berhasil! 🎉")

                            st.subheader("B.2. Visualisasi Volatilitas Prediksi NGARCH 📈")
                            fig_volatility = go.Figure()
                            fig_volatility.add_trace(go.Scatter(x=arima_residuals.index, y=predicted_volatility_series, mode='lines', name='Volatilitas Prediksi NGARCH', line=dict(color='#ff7f0e')))
                            fig_volatility.update_layout(title_text=f'Volatilitas Prediksi NGARCH ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                            st.plotly_chart(fig_volatility)

                            # --- Prediksi Volatilitas Masa Depan ---
                            st.subheader("B.3. Prediksi Volatilitas Masa Depan ⏩")
                            num_future_vol_steps = st.number_input("Jumlah langkah prediksi volatilitas ke depan:", min_value=1, max_value=30, value=7, key="num_future_vol_steps_ngarch")
                            
                            if st.button(f"Prediksi {num_future_vol_steps} Volatilitas ke Depan", key="predict_future_vol_button"):
                                with st.spinner(f"Memprediksi {num_future_vol_steps} volatilitas ke depan... ⏳"):
                                    try:
                                        # Prediksi volatilitas out-of-sample
                                        # Menggunakan forecast() method
                                        # start = len(arima_residuals) jika ingin forecast langsung setelah data pelatihan
                                        # Atau ambil tanggal terakhir dari data residual untuk memulai forecast
                                        last_date_residuals = arima_residuals.index[-1]
                                        
                                        # Buat indeks tanggal masa depan
                                        future_vol_dates = pd.date_range(start=last_date_residuals + pd.Timedelta(days=1), periods=num_future_vol_steps, freq='D')

                                        # Forecast menggunakan model_ngarch_fit.forecast(horizon=...)
                                        # forecast_results.variance.values[-1, :] memberikan prediksi varians untuk horizon yang ditentukan
                                        forecast_ngarch = st.session_state['model_ngarch_fit'].forecast(horizon=num_future_vol_steps, start=None) # start=None will use the end of the data

                                        # conditional_volatility adalah akar kuadrat dari varians
                                        future_predicted_volatility = np.sqrt(forecast_ngarch.variance.values[-1, :])
                                        future_predicted_volatility_series = pd.Series(future_predicted_volatility, index=future_vol_dates)

                                        st.session_state['future_predicted_volatility_series'] = future_predicted_volatility_series
                                        st.session_state['last_forecast_volatility_ngarch'] = future_predicted_volatility_series.iloc[-1]

                                        st.success(f"Prediksi volatilitas untuk {num_future_vol_steps} hari ke depan berhasil! 🎉")
                                        st.dataframe(future_predicted_volatility_series)

                                        fig_future_vol = go.Figure()
                                        fig_future_vol.add_trace(go.Scatter(x=predicted_volatility_series.index, y=predicted_volatility_series, mode='lines', name='Volatilitas Prediksi (Data Latih)', line=dict(color='#ff7f0e')))
                                        fig_future_vol.add_trace(go.Scatter(x=future_predicted_volatility_series.index, y=future_predicted_volatility_series, mode='lines', name='Volatilitas Prediksi (Masa Depan)', line=dict(color='#3d85c6', dash='solid')))
                                        fig_future_vol.update_layout(title_text=f'Prediksi Volatilitas NGARCH {st.session_state.get("selected_currency", "")} (Historis & Masa Depan)', xaxis_rangeslider_visible=True)
                                        st.plotly_chart(fig_future_vol)
                                        
                                    except Exception as e:
                                        st.error(f"Terjadi kesalahan saat memprediksi volatilitas masa depan: {e} ❌")
                            else:
                                st.info("Masukkan jumlah langkah dan klik tombol untuk memprediksi volatilitas masa depan. 🗓️")

                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat melakukan prediksi volatilitas: {e} ❌")
                else:
                    st.info("Klik 'Lakukan Prediksi Volatilitas' untuk melihat hasil prediksi dan visualisasi. ➡️")
            else:
                st.warning("Model NGARCH belum dilatih. Silakan latih model terlebih dahulu di bagian 'A.2. Latih Model NGARCH'. ⚠️")
        else:
            st.warning("Uji ARCH pada residual ARIMA tidak menunjukkan adanya efek heteroskedastisitas (varians konstan). Model NGARCH mungkin tidak diperlukan. Namun, Anda tetap dapat melatihnya jika ingin mengeksplorasi.")
            st.info("Jika Anda tetap ingin melatih model NGARCH, lanjutkan dengan memilih ordo dan klik tombol latihan.")
            # Still offer the option to train even if ARCH test is negative
            st.subheader("A. Pemodelan NGARCH (Variance Equation) ⚙️")
            p_ngarch = st.number_input("Ordo NGARCH (p):", min_value=1, max_value=5, value=1, key="ngarch_p_no_arch")
            q_ngarch = st.number_input("Ordo NGARCH (q):", min_value=1, max_value=5, value=1, key="ngarch_q_no_arch")
            if st.button("A.2. Latih Model NGARCH (Meski Tanpa ARCH Effect) ▶️", key="train_ngarch_button_no_arch"):
                with st.spinner("Melatih model NGARCH... ⏳"):
                    try:
                        model_ngarch = arch_model(arima_residuals, x=None, p=p_ngarch, o=1, q=q_ngarch, dist='normal', vol='NGARCH')
                        model_ngarch_fit = model_ngarch.fit(disp='off')

                        st.session_state['model_ngarch_fit'] = model_ngarch_fit
                        st.success("Model NGARCH berhasil dilatih! 🎉")

                        st.subheader("A.3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                        st.text(model_ngarch_fit.summary().as_text())
                        
                        st.subheader("A.4. Uji Signifikansi Koefisien NGARCH (P-value) ✅❌")
                        ngarch_summary_html = model_ngarch_fit.summary().tables[1].as_html()
                        df_ngarch_results = pd.read_html(ngarch_summary_html, header=0, index_col=0)[0]
                        st.dataframe(df_ngarch_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e} ❌")
        
            if st.session_state['model_ngarch_fit'] is not None:
                st.markdown("<h3 class='section-header'>B. Prediksi Volatilitas Menggunakan NGARCH 🌪️</h3>", unsafe_allow_html=True)
                if st.button("B.1. Lakukan Prediksi Volatilitas ▶️", key="predict_volatility_button_no_arch"):
                    with st.spinner("Memprediksi volatilitas dengan NGARCH... ⏳"):
                        try:
                            in_sample_forecast = st.session_state['model_ngarch_fit'].conditional_volatility
                            predicted_volatility_series = pd.Series(in_sample_forecast, index=arima_residuals.index)
                            st.session_state['predicted_volatility_series'] = predicted_volatility_series

                            st.success("Prediksi volatilitas in-sample berhasil! 🎉")

                            st.subheader("B.2. Visualisasi Volatilitas Prediksi NGARCH 📈")
                            fig_volatility = go.Figure()
                            fig_volatility.add_trace(go.Scatter(x=arima_residuals.index, y=predicted_volatility_series, mode='lines', name='Volatilitas Prediksi NGARCH', line=dict(color='#ff7f0e')))
                            fig_volatility.update_layout(title_text=f'Volatilitas Prediksi NGARCH ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                            st.plotly_chart(fig_volatility)

                            st.subheader("B.3. Prediksi Volatilitas Masa Depan ⏩")
                            num_future_vol_steps = st.number_input("Jumlah langkah prediksi volatilitas ke depan:", min_value=1, max_value=30, value=7, key="num_future_vol_steps_ngarch_no_arch")
                            if st.button(f"Prediksi {num_future_vol_steps} Volatilitas ke Depan", key="predict_future_vol_button_no_arch"):
                                with st.spinner(f"Memprediksi {num_future_vol_steps} volatilitas ke depan... ⏳"):
                                    try:
                                        last_date_residuals = arima_residuals.index[-1]
                                        future_vol_dates = pd.date_range(start=last_date_residuals + pd.Timedelta(days=1), periods=num_future_vol_steps, freq='D')
                                        forecast_ngarch = st.session_state['model_ngarch_fit'].forecast(horizon=num_future_vol_steps, start=None)
                                        future_predicted_volatility = np.sqrt(forecast_ngarch.variance.values[-1, :])
                                        future_predicted_volatility_series = pd.Series(future_predicted_volatility, index=future_vol_dates)

                                        st.session_state['future_predicted_volatility_series'] = future_predicted_volatility_series
                                        st.session_state['last_forecast_volatility_ngarch'] = future_predicted_volatility_series.iloc[-1]

                                        st.success(f"Prediksi volatilitas untuk {num_future_vol_steps} hari ke depan berhasil! 🎉")
                                        st.dataframe(future_predicted_volatility_series)

                                        fig_future_vol = go.Figure()
                                        fig_future_vol.add_trace(go.Scatter(x=predicted_volatility_series.index, y=predicted_volatility_series, mode='lines', name='Volatilitas Prediksi (Data Latih)', line=dict(color='#ff7f0e')))
                                        fig_future_vol.add_trace(go.Scatter(x=future_predicted_volatility_series.index, y=future_predicted_volatility_series, mode='lines', name='Volatilitas Prediksi (Masa Depan)', line=dict(color='#3d85c6', dash='solid')))
                                        fig_future_vol.update_layout(title_text=f'Prediksi Volatilitas NGARCH {st.session_state.get("selected_currency", "")} (Historis & Masa Depan)', xaxis_rangeslider_visible=True)
                                        st.plotly_chart(fig_future_vol)
                                    except Exception as e:
                                        st.error(f"Terjadi kesalahan saat memprediksi volatilitas masa depan: {e} ❌")
                            else:
                                st.info("Masukkan jumlah langkah dan klik tombol untuk memprediksi volatilitas masa depan. 🗓️")
                        except Exception as e:
                            st.error(f"Terjadi kesalahan saat melakukan prediksi volatilitas: {e} ❌")
                else:
                    st.info("Klik 'Lakukan Prediksi Volatilitas' untuk melihat hasil prediksi dan visualisasi. ➡️")
            else:
                st.warning("Model NGARCH belum dilatih. Silakan latih model terlebih dahulu di bagian 'A.2. Latih Model NGARCH'. ⚠️")

    else:
        st.warning("Residual ARIMA tidak tersedia untuk pemodelan NGARCH. Pastikan Anda telah melatih model ARIMA terlebih dahulu di halaman 'MODEL & PREDIKSI ARIMA'. ⚠️⬆️")

elif st.session_state['current_page'] == 'interpretasi_saran':
    st.markdown('<div class="main-header">INTERPRETASI & SARAN 💡</div>', unsafe_allow_html=True)
    st.write("Di halaman ini, Anda akan menemukan interpretasi dari hasil model ARIMA dan NGARCH serta saran-saran berdasarkan analisis. 🤔✨")

    st.subheader(f"Ringkasan Hasil untuk {st.session_state.get('selected_currency', 'Mata Uang')}")

    col_inter_1, col_inter_2 = st.columns(2)

    with col_inter_1:
        st.markdown("#### Hasil Prediksi Harga (ARIMA)")
        if st.session_state['rmse_price_arima'] is not None:
            st.metric("RMSE Harga ARIMA", f"{st.session_state['rmse_price_arima']:.4f}")
            st.metric("MAE Harga ARIMA", f"{st.session_state['mae_price_arima']:.4f}")
            st.metric("MAPE Harga ARIMA", f"{st.session_state['mape_price_arima']:.2f}%")
            if not st.session_state['future_predicted_prices_series'].empty:
                st.write(f"Harga prediksi terakhir: **{st.session_state['last_forecast_price_arima']:.4f}**")
                st.write("Prediksi Harga Masa Depan:")
                st.dataframe(st.session_state['future_predicted_prices_series'])
            else:
                st.info("Belum ada prediksi harga masa depan. Jalankan prediksi di bagian 'MODEL & PREDIKSI ARIMA'.")
        else:
            st.info("Model ARIMA belum dilatih atau prediksi belum dilakukan.")

    with col_inter_2:
        st.markdown("#### Hasil Prediksi Volatilitas (NGARCH)")
        if st.session_state['model_ngarch_fit'] is not None:
            if st.session_state['arima_residual_has_arch_effect'] is True:
                st.success("Ada ARCH Effect terdeteksi, pemodelan NGARCH relevan.")
            elif st.session_state['arima_residual_has_arch_effect'] is False:
                st.info("Tidak ada ARCH Effect terdeteksi, namun model NGARCH telah dilatih.")
            else:
                st.warning("Status ARCH Effect belum diuji.")

            if not st.session_state['future_predicted_volatility_series'].empty:
                st.write(f"Volatilitas prediksi terakhir: **{st.session_state['last_forecast_volatility_ngarch']:.4f}**")
                st.write("Prediksi Volatilitas Masa Depan:")
                st.dataframe(st.session_state['future_predicted_volatility_series'])
            else:
                st.info("Belum ada prediksi volatilitas masa depan. Jalankan prediksi di bagian 'MODEL & PREDIKSI NGARCH'.")
        else:
            st.info("Model NGARCH belum dilatih atau prediksi belum dilakukan.")

    st.markdown("#### Interpretasi Umum Hasil Model")

    if st.session_state['model_arima_fit'] is not None:
        st.markdown("##### Interpretasi Model ARIMA:")
        st.write("- **Signifikansi Koefisien:** Periksa P-value dari koefisien AR dan MA di ringkasan model ARIMA. Jika P-value < 0.05, koefisien tersebut signifikan secara statistik, artinya lag tersebut penting untuk memprediksi return.")
        st.write("- **Residual:**")
        if st.session_state.get('arima_residuals', pd.Series()).empty:
             st.info("Residual ARIMA belum tersedia.")
        else:
            lb_test_res = sm.stats.acorr_ljungbox(st.session_state['arima_residuals'], lags=[10], return_df=True)
            if lb_test_res['lb_pvalue'].iloc[0] < 0.05:
                st.warning("  - Uji Ljung-Box menunjukkan residual ARIMA **masih berkorelasi**. Ini berarti model ARIMA mungkin belum sepenuhnya menangkap pola dalam data return. Pertimbangkan ordo ARIMA lain atau model yang lebih kompleks.")
            else:
                st.success("  - Uji Ljung-Box menunjukkan residual ARIMA **sudah white noise**. Model ARIMA sudah cukup baik dalam memodelkan bagian mean dari deret waktu.")
            
            jb_test_res = stats.jarque_bera(st.session_state['arima_residuals'])
            if jb_test_res[1] < 0.05:
                st.warning("  - Uji Jarque-Bera menunjukkan residual ARIMA **tidak normal**. Ini sangat umum pada data keuangan, mengindikasikan adanya 'fat tails' atau 'leptokurtosis' (banyak outlier). Ini adalah alasan utama untuk menggunakan model GARCH/NGARCH untuk memodelkan volatilitasnya.")
            else:
                st.success("  - Uji Jarque-Bera menunjukkan residual ARIMA **berdistribusi normal**.")

            if st.session_state['arima_residual_has_arch_effect'] is True:
                st.warning("  - Uji ARCH menunjukkan residual ARIMA **memiliki efek ARCH (heteroskedastisitas)**. Ini adalah indikasi kuat bahwa varians return tidak konstan dan model NGARCH (atau keluarga GARCH lainnya) diperlukan untuk memodelkan volatilitas.")
            elif st.session_state['arima_residual_has_arch_effect'] is False:
                st.success("  - Uji ARCH menunjukkan residual ARIMA **tidak memiliki efek ARCH**. Varians return cenderung konstan.")
            else:
                st.info("  - Uji ARCH pada residual ARIMA belum dilakukan.")

    if st.session_state['model_ngarch_fit'] is not None:
        st.markdown("##### Interpretasi Model NGARCH:")
        st.write("- **Signifikansi Koefisien:** Periksa P-value dari koefisien `omega`, `alpha`, `beta`, dan `gamma` (jika ada) di ringkasan model NGARCH. Jika P-value < 0.05, koefisien tersebut signifikan, yang berarti ada efek volatilitas klustering (alpha, beta) dan asimetri (gamma).")
        st.write("- **Volatilitas Klustering:** Koefisien `alpha` (respon terhadap guncangan masa lalu) dan `beta` (persistensi volatilitas) yang signifikan menunjukkan bahwa periode volatilitas tinggi cenderung diikuti oleh periode volatilitas tinggi, dan sebaliknya.")
        st.write("- **Efek Asimetri (Leverage Effect):** Untuk model NGARCH, koefisien `gamma` yang signifikan dan positif menunjukkan bahwa guncangan negatif (berita buruk) memiliki dampak yang lebih besar pada volatilitas masa depan dibandingkan guncangan positif (berita baik) dengan besaran yang sama. Ini adalah fitur penting dari NGARCH.")
    
    st.markdown("#### Saran dan Rekomendasi 💡")
    st.markdown("""
    <ul>
        <li>Jika model ARIMA menunjukkan residual yang masih berkorelasi (Ljung-Box P-value < 0.05), pertimbangkan untuk mencoba kombinasi ordo (p,q) ARIMA yang berbeda atau model lain yang lebih canggih.</li>
        <li>Non-normalitas residual (Jarque-Bera P-value < 0.05) adalah hal yang diharapkan pada data keuangan. Model GARCH/NGARCH dirancang untuk menangani ini dengan memodelkan varians bersyarat.</li>
        <li>Jika ada ARCH effect (Uji ARCH P-value < 0.05), penggunaan model NGARCH sangat tepat untuk memodelkan dan memprediksi volatilitas.</li>
        <li>Evaluasi kinerja model (RMSE, MAE, MAPE) memberikan gambaran seberapa baik model memprediksi. Nilai yang lebih rendah menunjukkan kinerja yang lebih baik.</li>
        <li>Prediksi volatilitas dari NGARCH dapat digunakan untuk manajemen risiko, penetapan harga opsi, atau strategi trading berbasis volatilitas.</li>
    </ul>
    """, unsafe_allow_html=True)

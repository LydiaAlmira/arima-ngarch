import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler # Meskipun tidak digunakan untuk ARIMA/GARCH, tetap disertakan jika ada kebutuhan lain
from sklearn.model_selection import train_test_split # Meskipun tidak digunakan langsung untuk splitting time series
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm # Untuk Ljung-Box, Jarque-Bera
from scipy import stats # Untuk Jarque-Bera test
from arch.univariate import ARX, NGARCH, Normal, StudentsT, SkewStudent # Menggunakan NGARCH sesuai permintaan

# Impor model yang relevan
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

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
            box_shadow: 0 0 0 0.2rem rgba(90, 150, 250, 0.25);
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
    # Jika tidak, berikan pesan peringatan atau inisialisasi kosong
    train_data_returns = st.session_state.get('train_data_returns', pd.Series())
    test_data_returns = st.session_state.get('test_data_returns', pd.Series())
    original_prices_series = st.session_state.get('original_prices_for_reconstruction', pd.Series())
    return_type = st.session_state.get('return_type', "Log Return")

    if not train_data_returns.empty and not test_data_returns.empty and not original_prices_series.empty:
        # --- MODEL ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>A. Pemodelan ARIMA (Mean Equation) ⚙️</h3>", unsafe_allow_html=True)
        st.write(f"Data pelatihan return untuk pemodelan ARIMA ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_returns.head())

        st.subheader("A.1. Tentukan Ordo ARIMA (p, d, q) 🔢")
        st.info("Berdasarkan plot ACF dan PACF di bagian 'Stasioneritas Data', Anda dapat memperkirakan ordo (p, q). Ordo differencing (d) harus 0 karena Anda sudah bekerja dengan data return yang diharapkan stasioner.")
        p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
        d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=0, value=0, help="Untuk data return, 'd' harus 0 karena data sudah distasionerkan.", key="arima_d")
        q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

        if st.button("A.2. Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.success("Model ARIMA berhasil dilatih! 🎉")
                    st.subheader("A.3. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    # Solusi untuk AttributeError: 'ARIMAResultsWrapper' object has no attribute 'tables'
                    model_summary = model_arima_fit.summary()
                    results_table_html = model_summary.tables[1].as_html()  
                    df_results = pd.read_html(results_table_html, header=0, index_col=0)[0]
                    st.dataframe(df_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P >= 0.05)")

                    st.subheader("A.5. Uji Asumsi Residual Model ARIMA 📊")
                    arima_residuals = model_arima_fit.resid.dropna()
                    st.session_state['arima_residuals'] = arima_residuals # Simpan residual untuk NGARCH

                    if not arima_residuals.empty:
                        # Plot Residual
                        st.write("##### Plot Residual ARIMA")
                        fig_res = go.Figure()
                        fig_res.add_trace(go.Scatter(x=arima_residuals.index, y=arima_residuals, mode='lines', name='Residual ARIMA', line=dict(color='#4c78a8')))
                        fig_res.update_layout(title_text=f'Residual Model ARIMA ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_res)

                        # Uji Normalitas (Jarque-Bera)
                        st.write("##### Uji Normalitas (Jarque-Bera Test)")
                        jb_test = stats.jarque_bera(arima_residuals)
                        st.write(f"Statistik Jarque-Bera: {jb_test[0]:.4f}")
                        st.write(f"P-value: {jb_test[1]:.4f}")
                        if jb_test[1] > 0.05:
                            st.success("Residual **terdistribusi normal** (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual **tidak terdistribusi normal** (tolak H0). ⚠️ (umum untuk data keuangan)")
                            st.info("Ketidaknormalan residual sering terjadi pada data keuangan karena sifat *fat tails* dan *skewness*. Model GARCH dapat mengatasi hal ini.")

                        # Uji Autokorelasi (Ljung-Box Test)
                        st.write("##### Uji Autokorelasi (Ljung-Box Test)")
                        lb_test = sm.stats.acorr_ljungbox(arima_residuals, lags=[10], return_df=True) # Uji pada lags 10
                        st.write(lb_test)
                        if lb_test['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual **memiliki autokorelasi** signifikan (tolak H0). ⚠️ Ini menunjukkan model ARIMA mungkin belum menangkap semua pola.")
                            st.info("Jika ada autokorelasi, pertimbangkan ordo ARIMA yang berbeda atau model yang lebih kompleks.")

                        # Uji Heteroskedastisitas (ARCH Test - Ljung-Box pada residual kuadrat)
                        st.write("##### Uji Heteroskedastisitas (Ljung-Box Test pada Residual Kuadrat)")
                        lb_arch_test = sm.stats.acorr_ljungbox(arima_residuals**2, lags=[10], return_df=True) # Uji pada lags 10
                        st.write(lb_arch_test)
                        if lb_arch_test['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual **tidak memiliki efek ARCH/GARCH** signifikan (gagal tolak H0). ✅")
                            st.info("Jika tidak ada efek ARCH/GARCH, mungkin model NGARCH tidak diperlukan, atau residual ARIMA sudah sangat baik.")
                            st.session_state['arima_residual_has_arch_effect'] = False
                        else:
                            st.warning("Residual **memiliki efek ARCH/GARCH** signifikan (tolak H0). ⚠️ Ini menunjukkan adanya volatilitas kelompok, sehingga model GARCH/NGARCH cocok untuk residual ini.")
                            st.session_state['arima_residual_has_arch_effect'] = True

                    else:
                        st.warning("Residual ARIMA kosong atau hanya berisi NaN. Tidak dapat melakukan uji asumsi. ⚠️")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e} ❌")
                st.info("Pastikan data return Anda sesuai dan ordo ARIMA yang dipilih valid. Residual mungkin berisi NaN jika ada masalah konvergensi. Periksa juga data apakah ada nilai tak terbatas/NaN setelah preprocessing. ⚠️")
        else:
            st.info("Klik 'A.2. Latih Model ARIMA' untuk memulai pemodelan. ➡️")

        # --- PREDIKSI ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>B. Prediksi ARIMA (Nilai Tukar) 🔮</h3>", unsafe_allow_html=True)
        st.write(f"Lakukan prediksi nilai tukar menggunakan model ARIMA yang telah dilatih pada data {st.session_state.get('selected_currency', '')}. 📊")

        if 'model_arima_fit' in st.session_state:
            num_forecast_steps = st.number_input("B.1. Jumlah langkah prediksi ke depan (hari):", min_value=1, max_value=30, value=5, key="arima_num_forecast_steps")

            if st.button("B.2. Lakukan Prediksi dan Evaluasi ▶️", key="run_arima_prediction_button"):
                try:
                    with st.spinner("Melakukan prediksi ARIMA dan rekonstruksi harga... ⏳"):
                        model_arima_fit = st.session_state['model_arima_fit']
                        
                        # Predict in-sample on test data (returns)
                        # Use predict method with start and end to get predictions over test data range
                        # Ensure the predict output aligns with test_data_returns index
                        arima_forecast_returns_test = model_arima_fit.predict(start=test_data_returns.index.min(), end=test_data_returns.index.max(), typ='levels')
                        # Ensure it's a Series and indexed correctly
                        if isinstance(arima_forecast_returns_test, np.ndarray):
                            arima_forecast_returns_test = pd.Series(arima_forecast_returns_test, index=test_data_returns.index)
                        else: # If it's already a Series, ensure index matches (though predict should align if dates match)
                            arima_forecast_returns_test = arima_forecast_returns_test.reindex(test_data_returns.index)

                        # Generate future dates for out-of-sample forecast
                        if isinstance(original_prices_series.index, pd.DatetimeIndex):
                            last_date_full_data = original_prices_series.index.max()
                            future_dates = pd.date_range(start=last_date_full_data + pd.Timedelta(days=1), periods=num_forecast_steps, freq='D')
                        else:
                            last_idx_full_data = original_prices_series.index.max()
                            future_dates = pd.RangeIndex(start=last_idx_full_data + 1, stop=last_idx_full_data + 1 + num_forecast_steps)

                        # Forecast out-of-sample returns
                        forecast_out_of_sample_returns = model_arima_fit.forecast(steps=num_forecast_steps)
                        forecast_out_of_sample_returns.index = future_dates # Beri indeks tanggal masa depan

                        # --- Rekonstruksi Harga Asli dari Prediksi Return ---
                        # Rekonstruksi untuk data uji
                        # Ambil harga terakhir dari data training
                        last_train_price = original_prices_series.loc[train_data_returns.index[-1]]
                        predicted_prices_test_list = [last_train_price]

                        for r in arima_forecast_returns_test.values:
                            if return_type == "Log Return":
                                next_price = predicted_prices_test_list[-1] * np.exp(r)
                            else: # Simple Return
                                next_price = predicted_prices_test_list[-1] * (1 + r)
                            predicted_prices_test_list.append(next_price)
                        # Series untuk prediksi data uji (jangan sertakan harga awal)
                        predicted_prices_series = pd.Series(predicted_prices_test_list[1:], index=arima_forecast_returns_test.index)


                        # Rekonstruksi untuk prediksi masa depan
                        last_actual_price_full_data = original_prices_series.iloc[-1]
                        future_predicted_prices_list = [last_actual_price_full_data]

                        for r_future in forecast_out_of_sample_returns.values:
                            if return_type == "Log Return":
                                next_future_price = future_predicted_prices_list[-1] * np.exp(r_future)
                            else: # Simple Return
                                next_future_price = future_predicted_prices_list[-1] * (1 + r_future)
                            future_predicted_prices_list.append(next_future_price)
                        future_predicted_prices_series = pd.Series(future_predicted_prices_list[1:], index=future_dates)

                        st.success("Prediksi harga berhasil dilakukan! 🎉")

                        # --- Visualisasi Hasil Prediksi Nilai Tukar ---
                        st.subheader(f"B.3. Prediksi Nilai Tukar (ARIMA) untuk {st.session_state.get('selected_currency', '')} 📈")
                        fig_price = go.Figure()
                        fig_price.add_trace(go.Scatter(x=original_prices_series.index, y=original_prices_series.values, mode='lines', name='Harga Aktual', line=dict(color='#3f72af')))
                        fig_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Data Uji)', line=dict(color='#d62728', dash='dash')))
                        fig_price.add_trace(go.Scatter(x=future_predicted_prices_series.index, y=future_predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Masa Depan)', line=dict(color='#2ca02c', dash='dot')))
                        fig_price.update_layout(title_text=f'Prediksi Nilai Tukar ARIMA {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_price)

                        # --- Evaluasi Metrik (untuk data uji) ---
                        st.subheader("B.4. Metrik Evaluasi Prediksi Harga (Data Uji) 📊🔍")
                        # Pastikan indeks selaras sebelum menghitung metrik
                        # Ambil bagian dari original_prices_series yang sesuai dengan indeks prediksi data uji
                        actual_test_prices = original_prices_series.loc[predicted_prices_series.index]

                        if len(actual_test_prices) == len(predicted_prices_series) and not actual_test_prices.empty:
                            rmse_price = np.sqrt(mean_squared_error(actual_test_prices, predicted_prices_series))
                            mae_price = mean_absolute_error(actual_test_prices, predicted_prices_series)
                            
                            # MAPE requires non-zero actual values
                            # Filter out zero actual values to avoid division by zero
                            actual_test_prices_non_zero = actual_test_prices[actual_test_prices != 0]
                            predicted_prices_series_non_zero = predicted_prices_series[actual_test_prices != 0]

                            if not actual_test_prices_non_zero.empty:
                                mape_price = np.mean(np.abs((actual_test_prices_non_zero - predicted_prices_series_non_zero) / actual_test_prices_non_zero)) * 100
                            else:
                                mape_price = float('inf') # Indicate cannot compute MAPE if all actuals are zero

                            st.write(f"**Prediksi Nilai Tukar ({st.session_state.get('selected_currency', '')} pada data uji):**")
                            st.write(f"RMSE (Root Mean Squared Error): {rmse_price:.4f} 👇")
                            st.write(f"MAE (Mean Absolute Error): {mae_price:.4f} 👇")
                            st.write(f"MAPE (Mean Absolute Percentage Error): {mape_price:.2f}% 👇")
                            
                            st.session_state['rmse_price_arima'] = rmse_price
                            st.session_state['mae_price_arima'] = mae_price
                            st.session_state['mape_price_arima'] = mape_price
                        else:
                            st.warning("Ukuran data aktual dan prediksi tidak cocok untuk evaluasi harga pada data uji, atau data aktual kosong. Pastikan indeks dan panjangnya sesuai. ⚠️")

                        st.session_state['last_forecast_price_arima'] = future_predicted_prices_series.iloc[-1] if not future_predicted_prices_series.empty else None
                        st.session_state['future_predicted_prices_series'] = future_predicted_prices_series # Simpan untuk interpretasi
                        st.session_state['predicted_prices_series'] = predicted_prices_series # Simpan untuk interpretasi

                        # Opsi untuk mengunduh prediksi
                        forecast_df_to_save = pd.DataFrame({
                            f'Predicted_{st.session_state.get("selected_currency", "")}': future_predicted_prices_series
                        })
                        st.download_button(
                            label=f"Unduh Prediksi Harga {st.session_state.get('selected_currency', '')} sebagai CSV ⬇️",
                            data=forecast_df_to_save.to_csv().encode('utf-8'),
                            file_name=f'forecast_{st.session_state.get("selected_currency", "")}_arima_prices.csv',
                            mime='text/csv',
                        )

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melakukan prediksi ARIMA: {e} ❌")
                    st.info("Harap periksa kembali langkah 'MODEL ARIMA' atau ordo model yang dipilih. Pastikan model telah dilatih dan data tersedia. ⚠️")
            else:
                st.info("Klik 'B.2. Lakukan Prediksi dan Evaluasi' untuk melihat hasil prediksi harga. ➡️")
        else:
            st.info("Silakan latih model ARIMA di bagian A terlebih dahulu. ⬆️")
    else:
        st.info("Harap pastikan semua langkah sebelumnya (Input Data, Data Preprocessing, Data Splitting) telah selesai dan mata uang telah dipilih. ⬆️")

elif st.session_state['current_page'] == 'ngarch_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI NGARCH 🌪️🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual dari model ARIMA {st.session_state.get('selected_currency', '')} untuk memodelkan volatilitas (varians), lalu lakukan prediksi volatilitas. 📊")

    # Pastikan residual ARIMA tersedia dari halaman sebelumnya
    arima_residuals = st.session_state.get('arima_residuals', pd.Series())
    arima_residual_has_arch_effect = st.session_state.get('arima_residual_has_arch_effect', None)
    train_data_returns = st.session_state.get('train_data_returns', pd.Series())
    test_data_returns = st.session_state.get('test_data_returns', pd.Series())
    original_prices_series = st.session_state.get('original_prices_for_reconstruction', pd.Series())


    if not arima_residuals.empty:
        st.write(f"Data residual ARIMA untuk pemodelan NGARCH ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(arima_residuals.head())

        if arima_residual_has_arch_effect == False:
            st.warning("Uji Heteroskedastisitas pada residual ARIMA menunjukkan **TIDAK ADA EFEK ARCH/GARCH** signifikan. Ini berarti model NGARCH mungkin tidak diperlukan atau tidak akan memberikan peningkatan signifikan. Namun, Anda tetap bisa mencobanya. 🧐")
        elif arima_residual_has_arch_effect == True:
            st.success("Uji Heteroskedastisitas pada residual ARIMA menunjukkan **ADA EFEK ARCH/GARCH** signifikan. Model NGARCH sangat cocok untuk residual ini! ✅")
        else:
            st.info("Latih Model ARIMA terlebih dahulu untuk menjalankan uji asumsi residual.")


        st.subheader("A.1. Tentukan Ordo NGARCH (p, q) dan Distribusi Error 🔢")
        st.info("Ordo p dan q untuk NGARCH biasanya diturunkan dari pola ACF/PACF pada residual kuadrat, atau dimulai dengan (1,1). Distribusi error default adalah Normal. Jika residual tidak normal, coba StudentsT atau SkewStudent.")
        
        # Ordo GARCH
        ngarch_p = st.number_input("Ordo GARCH (p):", min_value=1, max_value=5, value=1, key="ngarch_p")
        ngarch_o = st.number_input("Ordo Asymmetry (o):", min_value=0, max_value=1, value=1, help="Untuk NGARCH, 'o' menunjukkan order asymetric effect. Biasanya 1 untuk efek tuas.", key="ngarch_o") # Ordo untuk efek asimetris (leverage)
        ngarch_q = st.number_input("Ordo GARCH (q):", min_value=1, max_value=5, value=1, key="ngarch_q")
        
        # Distribusi Error
        dist_options = {
            "Normal": Normal,
            "Student's T": StudentsT,
            "Skew Student's T": SkewStudent
        }
        selected_dist = st.selectbox("Pilih Distribusi Error:", list(dist_options.keys()), key="ngarch_dist")
        
        if st.button("A.2. Latih Model NGARCH ▶️", key="train_ngarch_button"):
            try:
                with st.spinner("Melatih model NGARCH... ⏳"):
                    # NGARCH expects the residuals from the mean model
                    # The ARX model in 'arch' package allows direct specification of mean model parameters
                    # However, since we already have ARIMA residuals, we fit ARCH/GARCH on them directly.
                    # The 'power' parameter for NGARCH is typically 2, for standard deviation.
                    # The 'o' parameter is for the asymmetry/leverage effect order.
                    
                    # Ensure residuals are Series with proper index for ARCH
                    if not isinstance(arima_residuals.index, pd.DatetimeIndex):
                         # If index is not DatetimeIndex, convert it or ensure proper frequency
                         st.warning("Indeks residual ARIMA bukan DatetimeIndex. Mencoba mengonversi atau mengatur frekuensi default. Pastikan data Anda berurutan waktu. ⚠️")
                         # For simplicity, if not datetime, arch will treat it as a simple index
                         # but for forecasting, proper frequency is better.
                         # A more robust solution might involve re-indexing with a dummy DatetimeIndex
                         # for the length of the residuals if original data was not date-indexed.
                         # For now, let's assume it can work without explicit DatetimeIndex for fitting if frequency is implicit.
                    
                    # Note: NGARCH in `arch` package has different parameterization than some other software.
                    # 'p' is for alpha (ARCH terms), 'o' is for asymmetry (leverage), 'q' is for beta (GARCH terms).
                    
                    model_ngarch = arch_model(
                        arima_residuals, 
                        p=ngarch_p, 
                        o=ngarch_o, 
                        q=ngarch_q, 
                        rescale=False, # Set to False if data is already small (e.g., returns)
                        volatility='NGARCH', # Specify NGARCH volatility
                        dist=selected_dist.lower().replace(" ", ""), # 'normal', 't', 'skewt'
                        mean='Zero' # Assuming residuals have zero mean after ARIMA fitting
                    )
                    
                    model_ngarch_fit = model_ngarch.fit(disp='off') # disp='off' suppresses detailed output during fitting

                    st.session_state['model_ngarch_fit'] = model_ngarch_fit
                    st.success("Model NGARCH berhasil dilatih! 🎉")
                    st.subheader("A.3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                    st.text(model_ngarch_fit.summary().as_text())

                    st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    # Ekstrak P-values dari ringkasan model
                    df_ngarch_results = model_ngarch_fit.summary().tables[1] # Usually the second table for parameters
                    df_ngarch_results_df = pd.read_html(df_ngarch_results.as_html(), header=0, index_col=0)[0]
                    st.dataframe(df_ngarch_results_df[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P >= 0.05)")

                    st.subheader("A.5. Uji Asumsi Residual Standarisasi Model NGARCH 📊")
                    # Squared standardized residuals should not have ARCH effects
                    standardized_residuals_squared = model_ngarch_fit.std_resid**2
                    
                    if not standardized_residuals_squared.empty:
                        # Plot Residual Standarisasi Kuadrat
                        st.write("##### Plot Residual Standarisasi Kuadrat NGARCH")
                        fig_std_res_sq = go.Figure()
                        fig_std_res_sq.add_trace(go.Scatter(x=standardized_residuals_squared.index, y=standardized_residuals_squared, mode='lines', name='Residual Standarisasi Kuadrat', line=dict(color='#8c564b')))
                        fig_std_res_sq.update_layout(title_text=f'Residual Standarisasi Kuadrat NGARCH ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_std_res_sq)

                        # Uji Normalitas (Jarque-Bera) pada residual standarisasi
                        st.write("##### Uji Normalitas (Jarque-Bera Test) pada Residual Standarisasi")
                        jb_test_std_res = stats.jarque_bera(model_ngarch_fit.std_resid.dropna())
                        st.write(f"Statistik Jarque-Bera: {jb_test_std_res[0]:.4f}")
                        st.write(f"P-value: {jb_test_std_res[1]:.4f}")
                        if jb_test_std_res[1] > 0.05:
                            st.success("Residual standarisasi **terdistribusi normal** (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual standarisasi **tidak terdistribusi normal** (tolak H0). ⚠️ Ini bisa mengindikasikan distribusi error yang salah atau sisa pola.")

                        # Uji Autokorelasi (Ljung-Box Test) pada residual standarisasi
                        st.write("##### Uji Autokorelasi (Ljung-Box Test) pada Residual Standarisasi")
                        lb_test_std_res = sm.stats.acorr_ljungbox(model_ngarch_fit.std_resid.dropna(), lags=[10], return_df=True)
                        st.write(lb_test_std_res)
                        if lb_test_std_res['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual standarisasi **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual standarisasi **memiliki autokorelasi** signifikan (tolak H0). ⚠️")
                            st.info("Ini mungkin menunjukkan bahwa ordo model NGARCH belum optimal atau ada pola yang belum tertangkap.")

                        # Uji Heteroskedastisitas (Ljung-Box Test pada residual standarisasi kuadrat)
                        st.write("##### Uji Heteroskedastisitas (Ljung-Box Test pada Residual Standarisasi Kuadrat)")
                        lb_arch_test_std_res_sq = sm.stats.acorr_ljungbox(standardized_residuals_squared.dropna(), lags=[10], return_df=True)
                        st.write(lb_arch_test_std_res_sq)
                        if lb_arch_test_std_res_sq['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual standarisasi kuadrat **tidak memiliki efek ARCH/GARCH** signifikan (gagal tolak H0). ✅")
                            st.info("Ini adalah indikasi baik bahwa model NGARCH telah berhasil menangkap dinamika volatilitas. 🎉")
                        else:
                            st.warning("Residual standarisasi kuadrat **masih memiliki efek ARCH/GARCH** signifikan (tolak H0). ⚠️")
                            st.info("Pertimbangkan ordo NGARCH yang berbeda, atau model volatilitas lainnya. 🧐")
                    else:
                        st.warning("Residual standarisasi NGARCH kosong atau hanya berisi NaN. Tidak dapat melakukan uji asumsi. ⚠️")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e} ❌")
                st.info("Pastikan residual ARIMA tidak kosong dan ordo NGARCH yang dipilih valid. Jika ada masalah konvergensi, coba ordo lain atau periksa data. ⚠️")
        else:
            st.info("Klik 'A.2. Latih Model NGARCH' untuk memulai pemodelan. ➡️")

    # --- PREDIKSI NGARCH SECTION ---
    st.markdown("<h3 class='section-header'>B. Prediksi Volatilitas (NGARCH) 🔮</h3>", unsafe_allow_html=True)
    st.write(f"Lakukan prediksi volatilitas menggunakan model NGARCH yang telah dilatih pada residual ARIMA {st.session_state.get('selected_currency', '')}. 📊")

    if 'model_ngarch_fit' in st.session_state and 'model_arima_fit' in st.session_state:
        num_forecast_steps_vol = st.number_input("B.1. Jumlah langkah prediksi volatilitas ke depan (hari):", min_value=1, max_value=30, value=5, key="ngarch_num_forecast_steps")

        if st.button("B.2. Lakukan Prediksi dan Evaluasi Volatilitas ▶️", key="run_ngarch_prediction_button"):
            try:
                with st.spinner("Melakukan prediksi volatilitas NGARCH... ⏳"):
                    model_ngarch_fit = st.session_state['model_ngarch_fit']
                    model_arima_fit = st.session_state['model_arima_fit'] # Diperlukan untuk indeks data asli

                    # Perhatikan: Prediksi ARCH/GARCH adalah untuk varians/standar deviasi, bukan harga
                    # Forecast test data volatility (in-sample forecast over the test period)
                    # Use model_ngarch_fit.conditional_volatility to get in-sample conditional volatility
                    
                    # To get predictions for the test set, we need to extend the model with the test residuals
                    # However, .forecast() method is designed for out-of-sample prediction.
                    # For in-sample on test set, we usually calculate conditional volatility on the *full* data,
                    # and then slice it for the test period.
                    
                    # Get conditional volatility for the entire dataset (training + test)
                    # We need the full set of original returns that were used to generate residuals
                    # This implies fitting ARX on original returns (not just residuals) or carefully
                    # managing index for conditional volatility which should align with original returns.
                    
                    # For simplicity, if we fit NGARCH on arima_residuals, then its conditional volatility
                    # will be for those residuals. If we want to predict 'returns' volatility, we need
                    # a full ARX-NGARCH setup.
                    
                    # Let's adjust this: The NGARCH model takes the *returns* as input, and its mean model
                    # is handled internally. We fit it on `arima_residuals` assuming they are `returns - mean_forecast`.
                    # To get conditional volatility for the test set, we calculate it over the whole range
                    # that was used to train ARIMA (to get residuals) and then slice it.
                    
                    # Reconstruct full residuals series (if sliced for training) or use the one from ARIMA
                    # If arima_residuals is only for training set, we need to re-generate it for full data
                    # Or, as typically done, NGARCH is fit on the full processed returns directly,
                    # with mean equation handled by ARX part.
                    
                    # Given the setup `model_ngarch = arch_model(arima_residuals, ...)`, `arima_residuals` are already
                    # the target for NGARCH.
                    
                    # Conditional volatility over the period of arima_residuals
                    conditional_volatility_full_data = model_ngarch_fit.conditional_volatility
                    
                    # Align with test data returns for evaluation
                    # The `test_data_returns` represents the actual returns for the test period.
                    # We need the *actual* historical volatility for the test set to compare against.
                    # Simple proxy: absolute returns or squared returns.
                    
                    # For evaluation, we compare predicted conditional volatility (from NGARCH fit)
                    # to the squared *actual* test returns (as a proxy for realized volatility).
                    # This needs careful index alignment.
                    
                    # Conditional volatility over the period of arima_residuals.
                    # We need to ensure conditional_volatility_full_data covers the test period.
                    # model_ngarch_fit.forecast() gives out-of-sample forecasts.
                    
                    # For in-sample conditional volatility, it's `model_ngarch_fit.conditional_volatility`.
                    # This will have the same index as `arima_residuals`.
                    # We need to map this to `test_data_returns` index.
                    
                    # If arima_residuals correspond to train_data_returns:
                    # Then conditional_volatility_full_data is effectively for train_data_returns
                    # We need to predict *new* volatility for test_data_returns' period.
                    
                    # Let's assume `arima_residuals` covers the training period from ARIMA model.
                    # We need to forecast volatilities for the `test_data_returns` period.
                    
                    # This requires feeding test_data_returns (or its residuals) into forecast.
                    # The `forecast` method of `arch` takes `horizon` (steps ahead).
                    # It also has `start` and `align`.
                    
                    # First, generate out-of-sample forecasts for the future
                    # (beyond the end of the original_prices_series)
                    
                    forecast_results = model_ngarch_fit.forecast(
                        horizon=num_forecast_steps_vol,
                        start=arima_residuals.index[-1] # Start from the last date of residuals
                    )
                    # The forecast gives 'variance' and 'h.cond' (conditional variance for the horizon)
                    # We want standard deviation (volatility)
                    future_predicted_volatility = np.sqrt(forecast_results.variance.iloc[-1]) # Take the last row, which contains the multi-step forecasts
                    
                    # Give it future dates
                    if isinstance(original_prices_series.index, pd.DatetimeIndex):
                        last_date_full_data = original_prices_series.index.max()
                        future_dates_vol = pd.date_range(start=last_date_full_data + pd.Timedelta(days=1), periods=num_forecast_steps_vol, freq='D')
                    else:
                        last_idx_full_data = original_prices_series.index.max()
                        future_dates_vol = pd.RangeIndex(start=last_idx_full_data + 1, stop=last_idx_full_data + 1 + num_forecast_steps_vol)
                    
                    future_predicted_volatility_series = pd.Series(
                        future_predicted_volatility.values,
                        index=future_dates_vol
                    )

                    # Now, for in-sample predictions for the test set:
                    # We need to get the conditional volatilities for the period of `test_data_returns`.
                    # The `model_ngarch_fit.conditional_volatility` already holds the in-sample volatility
                    # for the `arima_residuals` (which correspond to `train_data_returns`).
                    # To get predictions for `test_data_returns`, we need to forecast *from* the end of training.
                    
                    # We will use rolling forecast for the test set, which is more robust for evaluation.
                    # Or, simplify: just evaluate on the out-of-sample forecast for the test set's duration.
                    
                    # The `arch` library's `forecast` function can generate forecasts starting from a specific point.
                    # We need forecasts for the length of `test_data_returns`.
                    
                    # This is a bit tricky with `arch.forecast` for in-sample test period.
                    # A common approach is to refit the model on a rolling basis or use the `predict` method
                    # if it supports starting from an arbitrary point.
                    
                    # Let's simplify for demonstration: we'll use a forecast starting from the last training point
                    # covering the length of the test data.
                    
                    # The `arima_residuals` are from `train_data_returns`.
                    # To predict volatility for `test_data_returns`, we need to extend the forecast from the end of `train_data_returns`.
                    
                    # Start forecasting from the end of the `arima_residuals` (which is the end of training data)
                    forecast_test_period = model_ngarch_fit.forecast(
                        horizon=len(test_data_returns),
                        start=arima_residuals.index[-1] # Start from the last date of residuals (end of training)
                    )
                    predicted_volatility_test = np.sqrt(forecast_test_period.variance.iloc[-1]) # Take the last row for multi-step forecasts
                    
                    # Ensure the index for test period predictions is correct
                    predicted_volatility_test_series = pd.Series(
                        predicted_volatility_test.values,
                        index=test_data_returns.index # Align with actual test data dates
                    )


                    st.success("Prediksi volatilitas berhasil dilakukan! 🎉")

                    # --- Visualisasi Hasil Prediksi Volatilitas ---
                    st.subheader(f"B.3. Prediksi Volatilitas (NGARCH) untuk {st.session_state.get('selected_currency', '')} 📈")
                    fig_vol = go.Figure()
                    
                    # Plot historical actual volatility proxy (absolute returns)
                    # For visualization, we can plot actual absolute returns or squared returns
                    # as a proxy for realized volatility.
                    
                    # Use full processed returns for visualization
                    full_processed_returns = st.session_state['processed_returns']
                    actual_volatility_proxy = np.sqrt(full_processed_returns**2) # Absolute returns as proxy for volatility
                    
                    fig_vol.add_trace(go.Scatter(x=actual_volatility_proxy.index, y=actual_volatility_proxy.values, mode='lines', name='Volatilitas Aktual (Abs Return)', line=dict(color='#000000', dash='dot'))) # Darker for actual
                    
                    # Plot predicted volatility for test data
                    # It needs to be aligned with the test data period
                    # Ensure indices match for plotting on same timeline
                    
                    # Find the slice of actual_volatility_proxy that corresponds to the test data period
                    actual_volatility_proxy_test = actual_volatility_proxy.loc[predicted_volatility_test_series.index]
                    
                    fig_vol.add_trace(go.Scatter(x=predicted_volatility_test_series.index, y=predicted_volatility_test_series.values, mode='lines', name='Prediksi Volatilitas NGARCH (Data Uji)', line=dict(color='#ff9900', dash='dash'))) # Orange for test prediction
                    fig_vol.add_trace(go.Scatter(x=future_predicted_volatility_series.index, y=future_predicted_volatility_series.values, mode='lines', name='Prediksi Volatilitas NGARCH (Masa Depan)', line=dict(color='#0066cc', dash='dashdot'))) # Blue for future prediction
                    
                    fig_vol.update_layout(title_text=f'Prediksi Volatilitas NGARCH {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True,
                                        yaxis_title="Volatilitas (Standar Deviasi)")
                    st.plotly_chart(fig_vol)

                    # --- Evaluasi Metrik (untuk data uji) ---
                    st.subheader("B.4. Metrik Evaluasi Prediksi Volatilitas (Data Uji) 📊🔍")
                    # For evaluating volatility models, often compare predicted conditional volatility
                    # against realized volatility proxies (e.g., squared returns).
                    
                    # Use the squared actual returns from the test set as the 'actual' volatility
                    actual_test_squared_returns = test_data_returns**2
                    
                    # Ensure indices align for calculation
                    # The predicted_volatility_test_series is already indexed to test_data_returns
                    
                    if len(actual_test_squared_returns) == len(predicted_volatility_test_series) and not actual_test_squared_returns.empty:
                        # Compare predicted volatility (standard deviation) to actual absolute returns or squared returns.
                        # Using absolute returns for MAE/MAPE makes more intuitive sense for 'volatility'
                        # For RMSE, it's typically (variance_predicted - variance_actual)^2, so (vol_pred^2 - actual_ret^2)^2
                        
                        # Let's evaluate predicted volatility (std dev) against absolute returns
                        actual_test_abs_returns = np.abs(test_data_returns)
                        
                        rmse_vol = np.sqrt(mean_squared_error(actual_test_abs_returns, predicted_volatility_test_series))
                        mae_vol = mean_absolute_error(actual_test_abs_returns, predicted_volatility_test_series)
                        
                        # MAPE for volatility
                        actual_test_abs_returns_non_zero = actual_test_abs_returns[actual_test_abs_returns != 0]
                        predicted_volatility_test_series_non_zero = predicted_volatility_test_series[actual_test_abs_returns != 0]

                        if not actual_test_abs_returns_non_zero.empty:
                            mape_vol = np.mean(np.abs((actual_test_abs_returns_non_zero - predicted_volatility_test_series_non_zero) / actual_test_abs_returns_non_zero)) * 100
                        else:
                            mape_vol = float('inf') # Cannot compute MAPE if all actuals are zero

                        st.write(f"**Prediksi Volatilitas ({st.session_state.get('selected_currency', '')} pada data uji - dibandingkan dengan Absolute Return):**")
                        st.write(f"RMSE (Root Mean Squared Error): {rmse_vol:.4f} 👇")
                        st.write(f"MAE (Mean Absolute Error): {mae_vol:.4f} 👇")
                        st.write(f"MAPE (Mean Absolute Percentage Error): {mape_vol:.2f}% 👇")

                        st.session_state['rmse_vol_ngarch'] = rmse_vol
                        st.session_state['mae_vol_ngarch'] = mae_vol
                        st.session_state['mape_vol_ngarch'] = mape_vol

                    else:
                        st.warning("Ukuran data aktual dan prediksi tidak cocok untuk evaluasi volatilitas pada data uji, atau data aktual kosong. Pastikan indeks dan panjangnya sesuai. ⚠️")

                    st.session_state['last_forecast_volatility_ngarch'] = future_predicted_volatility_series.iloc[-1] if not future_predicted_volatility_series.empty else None
                    st.session_state['future_predicted_volatility_series'] = future_predicted_volatility_series # Simpan untuk interpretasi
                    st.session_state['predicted_volatility_series'] = predicted_volatility_test_series # Simpan untuk interpretasi

                    # Opsi untuk mengunduh prediksi
                    forecast_df_vol_to_save = pd.DataFrame({
                        f'Predicted_Volatility_{st.session_state.get("selected_currency", "")}': future_predicted_volatility_series
                    })
                    st.download_button(
                        label=f"Unduh Prediksi Volatilitas {st.session_state.get('selected_currency', '')} sebagai CSV ⬇️",
                        data=forecast_df_vol_to_save.to_csv().encode('utf-8'),
                        file_name=f'forecast_{st.session_state.get("selected_currency", "")}_ngarch_volatility.csv',
                        mime='text/csv',
                    )

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi NGARCH: {e} ❌")
                st.info("Harap periksa kembali langkah 'MODEL NGARCH' atau ordo model yang dipilih. Pastikan model telah dilatih dan residual tersedia. ⚠️")
        else:
            st.info("Klik 'B.2. Lakukan Prediksi dan Evaluasi Volatilitas' untuk melihat hasil prediksi volatilitas. ➡️")
    else:
        st.info("Silakan latih model NGARCH di bagian A terlebih dahulu, dan pastikan residual ARIMA sudah tersedia. ⬆️")

elif st.session_state['current_page'] == 'interpretasi_saran':
    st.markdown('<div class="main-header">INTERPRETASI & SARAN 💡</div>', unsafe_allow_html=True)
    st.write("Bagian ini memberikan interpretasi dari hasil model ARIMA dan NGARCH, serta saran untuk analisis lebih lanjut atau pengambilan keputusan. 🧠")

    st.subheader("A. Interpretasi Hasil Model ARIMA 📈")
    if 'model_arima_fit' in st.session_state and st.session_state['model_arima_fit'] is not None:
        model_arima_fit = st.session_state['model_arima_fit']
        st.markdown("<div class='interpretation-text'>", unsafe_allow_html=True)
        st.write(f"Model ARIMA({model_arima_fit.model.order[0]}, {model_arima_fit.model.order[1]}, {model_arima_fit.model.order[2]}) untuk {st.session_state.get('selected_currency', '')} telah dilatih.")
        st.write(f"**Ringkasan Model (P-value Koefisien):**")
        try:
            model_summary = model_arima_fit.summary()
            results_table_html = model_summary.tables[1].as_html()
            df_results = pd.read_html(results_table_html, header=0, index_col=0)[0]
            st.dataframe(df_results[['P>|z|']])
            st.write("Cermati kolom 'P>|z|'. Koefisien dengan P-value < 0.05 menunjukkan signifikansi statistik. Koefisien signifikan berkontribusi pada prediksi rata-rata return.")
        except Exception as e:
            st.warning(f"Tidak dapat menampilkan ringkasan koefisien ARIMA: {e}")

        st.write("**Kualitas Prediksi Harga (Data Uji):**")
        if st.session_state.get('rmse_price_arima') is not None:
            st.write(f"- RMSE (Root Mean Squared Error): {st.session_state['rmse_price_arima']:.4f}")
            st.write(f"- MAE (Mean Absolute Error): {st.session_state['mae_price_arima']:.4f}")
            if st.session_state['mape_price_arima'] != float('inf'):
                st.write(f"- MAPE (Mean Absolute Percentage Error): {st.session_state['mape_price_arima']:.2f}%")
            else:
                st.write("- MAPE: Tidak dapat dihitung (terdapat nilai nol pada harga aktual data uji).")
            st.info("Nilai RMSE, MAE, dan MAPE yang lebih rendah menunjukkan akurasi prediksi harga yang lebih baik. Namun, selalu bandingkan dengan benchmark atau ekspektasi domain.")
        else:
            st.info("Belum ada metrik evaluasi ARIMA yang tersedia. Lakukan prediksi ARIMA terlebih dahulu.")

        if not st.session_state.get('future_predicted_prices_series', pd.Series()).empty:
            st.write(f"**Prediksi Harga Masa Depan:**")
            st.dataframe(st.session_state['future_predicted_prices_series'])
            st.write(f"Harga {st.session_state.get('selected_currency', '')} yang diprediksi untuk langkah terakhir adalah **{st.session_state['last_forecast_price_arima']:.4f}**.")
            st.info("Prediksi harga ini memberikan gambaran tentang tren nilai tukar di masa depan berdasarkan pola historis.")
        else:
            st.info("Belum ada prediksi harga masa depan yang tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Model ARIMA belum dilatih. Silakan kunjungi halaman 'MODEL & PREDIKSI ARIMA'. ⬆️")


    st.subheader("B. Interpretasi Hasil Model NGARCH 🌪️")
    if 'model_ngarch_fit' in st.session_state and st.session_state['model_ngarch_fit'] is not None:
        model_ngarch_fit = st.session_state['model_ngarch_fit']
        st.markdown("<div class='interpretation-text'>", unsafe_allow_html=True)
        st.write(f"Model NGARCH({model_ngarch_fit.num_arch_lags}, {model_ngarch_fit.num_leverage_lags}, {model_ngarch_fit.num_garch_lags}) dengan distribusi {model_ngarch_fit.distribution.name} telah dilatih pada residual ARIMA.")
        
        st.write(f"**Ringkasan Model (P-value Koefisien):**")
        try:
            df_ngarch_results = model_ngarch_fit.summary().tables[1]
            df_ngarch_results_df = pd.read_html(df_ngarch_results.as_html(), header=0, index_col=0)[0]
            st.dataframe(df_ngarch_results_df[['P>|z|']])
            st.write("Cermati kolom 'P>|z|'. Koefisien dengan P-value < 0.05 menunjukkan signifikansi statistik. Koefisien signifikan menunjukkan adanya efek ARCH (volatilitas masa lalu), GARCH (volatilitas varians masa lalu), dan Asimetri (NGARCH).")
            
            # Interpretasi Koefisien NGARCH (contoh)
            params = model_ngarch_fit.params
            st.write("**Interpretasi Koefisien Utama NGARCH:**")
            if 'mu' in params: # If mean model is not 'Zero'
                st.write(f"- **Mu (Rata-rata residual):** {params['mu']:.4f}. Idealnya mendekati nol.")
            if 'omega' in params:
                st.write(f"- **Omega (Konstanta varians):** {params['omega']:.4f}. Ini adalah varians dasar jangka panjang.")
            for i in range(model_ngarch_fit.num_arch_lags):
                if f'alpha[{i+1}]' in params:
                    st.write(f"- **Alpha[{i+1}] (ARCH term):** {params[f'alpha[{i+1}]']:.4f}. Mengukur respons volatilitas terhadap shock masa lalu. Nilai positif menunjukkan volatilitas meningkat setelah shock besar.")
            for i in range(model_ngarch_fit.num_garch_lags):
                if f'beta[{i+1}]' in params:
                    st.write(f"- **Beta[{i+1}] (GARCH term):** {params[f'beta[{i+1}]']:.4f}. Mengukur persistensi volatilitas. Nilai tinggi menunjukkan volatilitas akan bertahan dalam periode yang lama.")
            for i in range(model_ngarch_fit.num_leverage_lags):
                if f'gamma[{i+1}]' in params:
                    st.write(f"- **Gamma[{i+1}] (Asymmetry/Leverage term):** {params[f'gamma[{i+1}]']:.4f}. Ini adalah karakteristik utama NGARCH. Nilai positif menunjukkan efek tuas (leverage effect), di mana shock negatif (penurunan harga) memiliki dampak yang lebih besar pada volatilitas masa depan daripada shock positif (kenaikan harga) dengan besaran yang sama. Semakin besar gamma, semakin kuat efek asimetrisnya.")

        except Exception as e:
            st.warning(f"Tidak dapat menampilkan ringkasan koefisien NGARCH: {e}")

        st.write("**Kualitas Prediksi Volatilitas (Data Uji):**")
        if st.session_state.get('rmse_vol_ngarch') is not None:
            st.write(f"- RMSE (Root Mean Squared Error): {st.session_state['rmse_vol_ngarch']:.4f}")
            st.write(f"- MAE (Mean Absolute Error): {st.session_state['mae_vol_ngarch']:.4f}")
            if st.session_state['mape_vol_ngarch'] != float('inf'):
                st.write(f"- MAPE (Mean Absolute Percentage Error): {st.session_state['mape_vol_ngarch']:.2f}%")
            else:
                st.write("- MAPE: Tidak dapat dihitung (terdapat nilai nol pada volatilitas aktual data uji).")
            st.info("Nilai RMSE, MAE, dan MAPE yang lebih rendah menunjukkan akurasi prediksi volatilitas yang lebih baik. Ini menunjukkan seberapa baik model menangkap fluktuasi pasar.")
        else:
            st.info("Belum ada metrik evaluasi NGARCH yang tersedia. Lakukan prediksi NGARCH terlebih dahulu.")

        if not st.session_state.get('future_predicted_volatility_series', pd.Series()).empty:
            st.write(f"**Prediksi Volatilitas Masa Depan:**")
            st.dataframe(st.session_state['future_predicted_volatility_series'])
            st.write(f"Volatilitas {st.session_state.get('selected_currency', '')} yang diprediksi untuk langkah terakhir adalah **{st.session_state['last_forecast_volatility_ngarch']:.4f}** (standar deviasi).")
            st.info("Prediksi volatilitas ini penting untuk manajemen risiko dan penetapan harga opsi, karena menunjukkan seberapa besar fluktuasi harga yang diharapkan di masa depan.")
        else:
            st.info("Belum ada prediksi volatilitas masa depan yang tersedia.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Model NGARCH belum dilatih. Silakan kunjungi halaman 'MODEL & PREDIKSI NGARCH'. ⬆️")


    st.subheader("C. Saran dan Rekomendasi Umum 🛠️")
    st.markdown("""
    <div class="guidance-list">
    <ul>
        <li>**Pemilihan Ordo Model:** Ordo ARIMA (p, d, q) dan NGARCH (p, o, q) sangat mempengaruhi performa model. Jika metrik evaluasi (RMSE, MAE, MAPE) tidak memuaskan, pertimbangkan untuk mencoba kombinasi ordo yang berbeda. Plot ACF/PACF adalah panduan awal, tetapi validasi silang (cross-validation) atau kriteria informasi (AIC/BIC) dapat membantu optimasi.</li>
        <li>**Uji Asumsi Residual:** Pastikan residual model ARIMA mendekati white noise dan tidak memiliki autokorelasi. Jika masih ada autokorelasi, ordo ARIMA mungkin perlu disesuaikan. Untuk model NGARCH, periksa residual standar yang dikuadratkan untuk memastikan tidak ada efek ARCH/GARCH yang tersisa.</li>
        <li>**Distribusi Error:** Jika uji Jarque-Bera pada residual standar NGARCH menunjukkan non-normalitas yang signifikan, mencoba distribusi error Student's T atau Skew Student's T di model NGARCH seringkali lebih sesuai untuk data keuangan yang cenderung memiliki *fat tails*.</li>
        <li>**Keterbatasan Model:** Model ARIMA dan NGARCH bersifat univariat, artinya hanya mempertimbangkan satu variabel. Untuk analisis yang lebih komprehensif, model multivariat (misalnya VAR-GARCH) yang mempertimbangkan hubungan antar mata uang mungkin diperlukan.</li>
        <li>**Data Aktual vs. Prediksi:** Selalu visualisasikan dan bandingkan prediksi dengan data aktual. Perhatikan periode volatilitas tinggi atau rendah, dan apakah model dapat menangkap perubahan tersebut.</li>
        <li>**Overfitting:** Berhati-hatilah terhadap overfitting, terutama jika ordo model terlalu tinggi atau jumlah data pelatihan terbatas. Lakukan validasi pada data uji yang tidak terlihat oleh model saat pelatihan.</li>
        <li>**Interpretasi Bisnis:** Terjemahkan hasil statistik ke dalam implikasi bisnis. Prediksi harga memberikan arah, sementara prediksi volatilitas menginformasikan risiko.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

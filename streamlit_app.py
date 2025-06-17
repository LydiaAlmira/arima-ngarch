import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import math
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm # Untuk Ljung-Box, Jarque-Bera
from scipy import stats # Untuk Jarque-Bera test

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
    "MODEL ARIMA 📈": "pemodelan_arima", # Diubah namanya
    "PREDIKSI ARIMA 📈": "prediksi_arima", # Tambahan menu prediksi untuk ARIMA
    "MODEL NGARCH 🌪️": "pemodelan_ngarch", # Diubah namanya
    "PREDIKSI NGARCH 🌪️": "prediksi_ngarch", # Tambahan menu prediksi untuk NGARCH
    "INTERPRETASI & SARAN 💡": "interpretasi_saran",
}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'selected_currency' not in st.session_state:
    st.session_state['selected_currency'] = None
if 'variable_name' not in st.session_state:
    st.session_state['variable_name'] = "Nama Variabel"

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
        <li><b>MODEL ARIMA 📈:</b> Langkah-langkah untuk membentuk model ARIMA pada data return (untuk prediksi nilai tukar), termasuk uji asumsi dan koefisien.</li>
        <li><b>PREDIKSI ARIMA 📈:</b> Menampilkan hasil prediksi nilai tukar dari model ARIMA dan evaluasinya.</li>
        <li><b>MODEL NGARCH 🌪️:</b> Langkah-langkah untuk membentuk model NGARCH pada residual ARIMA (untuk prediksi volatilitas), termasuk uji asumsi dan koefisien.</li>
        <li><b>PREDIKSI NGARCH 🌪️:</b> Menampilkan hasil prediksi volatilitas dari model NGARCH dan visualisasinya.</li>
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
                    st.write(f"  {key}: {value:.4f}")

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


elif st.session_state['current_page'] == 'pemodelan_arima':
    st.markdown('<div class="main-header">MODEL ARIMA (Mean Equation) ⚙️📈</div>', unsafe_allow_html=True)
    st.write(f"Latih model ARIMA pada data return {st.session_state.get('selected_currency', '')} untuk memodelkan mean (prediksi nilai tukar). 📊")

    if 'train_data_returns' in st.session_state and not st.session_state['train_data_returns'].empty:
        train_data_returns = st.session_state['train_data_returns']
        st.write(f"Data pelatihan return untuk pemodelan ARIMA ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_returns.head())

        st.subheader("1. Tentukan Ordo ARIMA (p, d, q) 🔢")
        st.info("Berdasarkan plot ACF dan PACF di bagian 'Stasioneritas Data', Anda dapat memperkirakan ordo (p, q). Ordo differencing (d) harus 0 karena Anda sudah bekerja dengan data return yang diharapkan stasioner.")
        p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
        d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=0, value=0, help="Untuk data return, 'd' harus 0 karena data sudah distasionerkan.", key="arima_d")
        q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

        if st.button("2. Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.success("Model ARIMA berhasil dilatih! 🎉")
                    st.subheader("3. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    results_table = model_arima_fit.summary().tables[1]
                    df_results = pd.read_html(results_table.as_html(), header=0, index_col=0)[0]
                    st.dataframe(df_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P >= 0.05)")

                    st.subheader("5. Uji Asumsi Residual Model ARIMA 📊")
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
        st.info("Silakan unggah, proses, dan bagi data terlebih dahulu di halaman 'Input Data', 'Data Preprocessing', dan 'Data Splitting'. ⬆️")

elif st.session_state['current_page'] == 'prediksi_arima':
    st.markdown('<div class="main-header">PREDIKSI ARIMA (Nilai Tukar) 🔮📈</div>', unsafe_allow_html=True)
    st.write(f"Lakukan prediksi nilai tukar menggunakan model ARIMA yang telah dilatih pada data {st.session_state.get('selected_currency', '')}. 📊")

    if 'model_arima_fit' in st.session_state and \
       'train_data_returns' in st.session_state and 'test_data_returns' in st.session_state and \
       'original_prices_for_reconstruction' in st.session_state:

        model_arima_fit = st.session_state['model_arima_fit']
        train_data_returns = st.session_state['train_data_returns']
        test_data_returns = st.session_state['test_data_returns']
        original_prices_series = st.session_state['original_prices_for_reconstruction']
        return_type = st.session_state.get('return_type', 'Log Return')

        st.subheader("1. Konfigurasi Prediksi ⚙️")
        num_forecast_steps = st.number_input("Jumlah langkah prediksi ke depan (hari):", min_value=1, max_value=30, value=5, key="arima_num_forecast_steps")

        if st.button("2. Lakukan Prediksi dan Evaluasi ▶️", key="run_arima_prediction_button"):
            try:
                with st.spinner("Melakukan prediksi ARIMA dan rekonstruksi harga... ⏳"):
                    # Predict in-sample on test data
                    # Start prediksi dari indeks setelah data latih terakhir
                    start_pred_idx_test = test_data_returns.index.min()
                    end_pred_idx_test = test_data_returns.index.max()

                    # Pastikan start dan end indeks ada di original_prices_series
                    # Karena ARIMA memprediksi *return*, kita butuh harga sebelumnya untuk merekonstruksi
                    # Kita akan memprediksi return untuk periode test_data_returns
                    arima_forecast_returns_test = model_arima_fit.predict(start=start_pred_idx_test, end=end_pred_idx_test, typ='levels')
                    arima_forecast_returns_test = arima_forecast_returns_test.reindex(test_data_returns.index) # Sesuaikan indeks

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
                    last_train_price = original_prices_series.loc[train_data_returns.index[-1]]
                    predicted_prices_test = [last_train_price]

                    for r in arima_forecast_returns_test.values:
                        if return_type == "Log Return":
                            next_price = predicted_prices_test[-1] * np.exp(r)
                        else:
                            next_price = predicted_prices_test[-1] * (1 + r)
                        predicted_prices_test.append(next_price)
                    predicted_prices_series = pd.Series(predicted_prices_test[1:], index=arima_forecast_returns_test.index)

                    # Rekonstruksi untuk prediksi masa depan
                    last_actual_price_full_data = original_prices_series.iloc[-1]
                    future_predicted_prices_list = [last_actual_price_full_data]

                    for r_future in forecast_out_of_sample_returns.values:
                        if return_type == "Log Return":
                            next_future_price = future_predicted_prices_list[-1] * np.exp(r_future)
                        else:
                            next_future_price = future_predicted_prices_list[-1] * (1 + r_future)
                        future_predicted_prices_list.append(next_future_price)
                    future_predicted_prices_series = pd.Series(future_predicted_prices_list[1:], index=future_dates)

                    st.success("Prediksi harga berhasil dilakukan! 🎉")

                    # --- Visualisasi Hasil Prediksi Nilai Tukar ---
                    st.subheader(f"3. Prediksi Nilai Tukar (ARIMA) untuk {st.session_state.get('selected_currency', '')} 📈")
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=original_prices_series.index, y=original_prices_series.values, mode='lines', name='Harga Aktual', line=dict(color='#3f72af')))
                    fig_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Data Uji)', line=dict(color='#d62728', dash='dash')))
                    fig_price.add_trace(go.Scatter(x=future_predicted_prices_series.index, y=future_predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Masa Depan)', line=dict(color='#2ca02c', dash='dot')))
                    fig_price.update_layout(title_text=f'Prediksi Nilai Tukar ARIMA {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_price)

                    # --- Evaluasi Metrik (untuk data uji) ---
                    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

                    st.subheader("4. Metrik Evaluasi Prediksi Harga (Data Uji) 📊🔍")
                    actual_test_prices = original_prices_series.loc[predicted_prices_series.index]

                    if len(actual_test_prices) == len(predicted_prices_series):
                        rmse_price = np.sqrt(mean_squared_error(actual_test_prices, predicted_prices_series))
                        mae_price = mean_absolute_error(actual_test_prices, predicted_prices_series)
                        mape_price = mean_absolute_percentage_error(actual_test_prices, predicted_prices_series) * 100 # dalam persentase

                        st.write(f"**Prediksi Nilai Tukar ({st.session_state.get('selected_currency', '')} pada data uji):**")
                        st.write(f"RMSE (Root Mean Squared Error): {rmse_price:.4f} 👇")
                        st.write(f"MAE (Mean Absolute Error): {mae_price:.4f} 👇")
                        st.write(f"MAPE (Mean Absolute Percentage Error): {mape_price:.2f}% 👇")
                        st.session_state['rmse_price_arima'] = rmse_price
                        st.session_state['mae_price_arima'] = mae_price
                        st.session_state['mape_price_arima'] = mape_price
                    else:
                        st.warning("Ukuran data aktual dan prediksi tidak cocok untuk evaluasi harga pada data uji. Pastikan indeks dan panjangnya sesuai. ⚠️")

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
        st.info("Harap pastikan semua langkah sebelumnya (Input Data, Data Preprocessing, Data Splitting, MODEL ARIMA) telah selesai dan mata uang telah dipilih. ⬆️")


elif st.session_state['current_page'] == 'pemodelan_ngarch':
    st.markdown('<div class="main-header">MODEL NGARCH (Volatilitas) 🌪️📊</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual dari model ARIMA untuk memodelkan volatilitas (varians bersyarat) untuk {st.session_state.get('selected_currency', '')}. 📉📈")

    if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
        arima_residuals = st.session_state['arima_residuals'].dropna()
        st.write(f"Residual dari model ARIMA ({st.session_state.get('selected_currency', '')}) (data untuk model NGARCH):")
        st.dataframe(arima_residuals.head())

        if arima_residuals.empty:
            st.warning("Residual ARIMA kosong atau hanya berisi NaN. Pastikan model ARIMA berhasil dilatih dan menghasilkan residual yang valid. ⚠️")
        else:
            st.subheader("1. Tentukan Ordo NGARCH (p, o, q) dan Distribusi Error 🔢")
            st.info("Pilih ordo p, o, q untuk model GARCH (p: ARCH order, o: asymmetry order, q: GARCH order). Ordo 'o' menangkap efek asimetris (leverage effect).")
            p_garch = st.number_input("Ordo ARCH (p):", min_value=1, max_value=3, value=1, key="ngarch_p")
            o_garch = st.number_input("Ordo Asymmetry (o):", min_value=0, max_value=2, value=1, key="ngarch_o")
            q_garch = st.number_input("Ordo GARCH (q):", min_value=1, max_value=3, value=1, key="ngarch_q")
            dist_garch = st.selectbox("Pilih Distribusi Error: 📉", ["normal", "t", "skewt"], index=1, key="ngarch_dist")

            if st.button("2. Latih Model NGARCH ▶️", key="train_ngarch_button"):
                try:
                    with st.spinner("Melatih model NGARCH... ⏳"):
                        model_ngarch = arch_model(arima_residuals, vol='Garch', p=p_garch, o=o_garch, q=q_garch, dist=dist_garch)
                        res_ngarch = model_ngarch.fit(disp='off')

                        st.session_state['model_ngarch_fit'] = res_ngarch
                        st.success("Model NGARCH (GJR-GARCH) berhasil dilatih! 🎉")
                        st.subheader("3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                        st.text(res_ngarch.summary().as_text())

                        st.subheader("4. Uji Signifikansi Koefisien (P-value) ✅❌")
                        st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05.")
                        
                        # Mengambil tabel koefisien dari summary arch_model
                        # arch_model summary sedikit berbeda, kita perlu parse secara manual atau gunakan .params dan .pvalues
                        params_df = pd.DataFrame({
                            'Parameter': res_ngarch.params.index,
                            'Koefisien': res_ngarch.params.values,
                            'P-value': res_ngarch.pvalues.values
                        })
                        st.dataframe(params_df[['Parameter', 'P-value']].style.applymap(lambda x: 'background-color: #d4edda' if isinstance(x, (int, float)) and x < 0.05 else 'background-color: #f8d7da', subset=['P-value']))
                        st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P >= 0.05)")

                        st.subheader("5. Uji Asumsi Residual Baku Model NGARCH 📊")
                        # Residual baku = residual / volatilitas kondisional
                        std_residuals = res_ngarch.resid / res_ngarch.conditional_volatility
                        st.session_state['ngarch_std_residuals'] = std_residuals.dropna()

                        if not std_residuals.dropna().empty:
                            # Plot Residual Baku
                            st.write("##### Plot Residual Baku NGARCH")
                            fig_std_res = go.Figure()
                            fig_std_res.add_trace(go.Scatter(x=std_residuals.index, y=std_residuals, mode='lines', name='Residual Baku NGARCH', line=dict(color='#8c564b')))
                            fig_std_res.update_layout(title_text=f'Residual Baku Model NGARCH ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                            st.plotly_chart(fig_std_res)

                            # Uji Normalitas (Jarque-Bera)
                            st.write("##### Uji Normalitas (Jarque-Bera Test) pada Residual Baku")
                            jb_test_std = stats.jarque_bera(std_residuals.dropna())
                            st.write(f"Statistik Jarque-Bera: {jb_test_std[0]:.4f}")
                            st.write(f"P-value: {jb_test_std[1]:.4f}")
                            if jb_test_std[1] > 0.05:
                                st.success("Residual baku **terdistribusi normal** (gagal tolak H0). ✅ (jarang untuk data keuangan)")
                            else:
                                st.warning("Residual baku **tidak terdistribusi normal** (tolak H0). ⚠️ Ini cukup umum untuk data keuangan.")
                                st.info("Model GARCH dengan distribusi t-Student atau skew-t dapat membantu menangani *fat tails* dan *skewness*.")

                            # Uji Autokorelasi (Ljung-Box Test) pada residual baku
                            st.write("##### Uji Autokorelasi (Ljung-Box Test) pada Residual Baku")
                            lb_std_test = sm.stats.acorr_ljungbox(std_residuals.dropna(), lags=[10], return_df=True)
                            st.write(lb_std_test)
                            if lb_std_test['lb_pvalue'].iloc[0] > 0.05:
                                st.success("Residual baku **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅")
                            else:
                                st.warning("Residual baku **memiliki autokorelasi** signifikan (tolak H0). ⚠️ Ini menunjukkan model mungkin belum sepenuhnya menangkap dependensi mean.")

                            # Uji Autokorelasi (Ljung-Box Test) pada residual baku kuadrat
                            st.write("##### Uji Autokorelasi (Ljung-Box Test) pada Residual Baku Kuadrat (untuk sisa efek ARCH)")
                            lb_std_sq_test = sm.stats.acorr_ljungbox(std_residuals.dropna()**2, lags=[10], return_df=True)
                            st.write(lb_std_sq_test)
                            if lb_std_sq_test['lb_pvalue'].iloc[0] > 0.05:
                                st.success("Residual baku kuadrat **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅")
                                st.info("Ini adalah indikasi bahwa model GARCH/NGARCH telah berhasil menangkap semua efek ARCH/GARCH dalam data.")
                            else:
                                st.warning("Residual baku kuadrat **memiliki autokorelasi** signifikan (tolak H0). ⚠️ Ini menunjukkan model mungkin belum sepenuhnya menangkap volatilitas kelompok (efek ARCH/GARCH).")
                                st.info("Pertimbangkan ordo GARCH yang berbeda atau model volatilitas lain.")

                        else:
                            st.warning("Residual baku NGARCH kosong atau hanya berisi NaN. Tidak dapat melakukan uji asumsi. ⚠️")


                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e} ❌")
                    st.info("Pastikan residual ARIMA valid dan ordo NGARCH yang dipilih tepat. Mungkin ada masalah konvergensi. Coba ganti distribusi error jika model tidak konvergen. ⚠️")
    else:
        st.info("Latih model ARIMA terlebih dahulu di halaman 'MODEL ARIMA' untuk mendapatkan residual. ⬆️")

elif st.session_state['current_page'] == 'prediksi_ngarch':
    st.markdown('<div class="main-header">PREDIKSI NGARCH (Volatilitas) 🔮🌪️</div>', unsafe_allow_html=True)
    st.write(f"Lakukan prediksi volatilitas menggunakan model NGARCH yang telah dilatih pada data {st.session_state.get('selected_currency', '')}. ✨")

    if 'model_ngarch_fit' in st.session_state and \
       'train_data_returns' in st.session_state and 'test_data_returns' in st.session_state:

        model_ngarch_fit = st.session_state['model_ngarch_fit']
        train_data_returns = st.session_state['train_data_returns']
        test_data_returns = st.session_state['test_data_returns']
        arima_residuals = st.session_state['arima_residuals'].dropna()

        st.subheader("1. Konfigurasi Prediksi Volatilitas ⚙️")
        num_forecast_steps_vol = st.number_input("Jumlah langkah prediksi volatilitas ke depan (hari):", min_value=1, max_value=30, value=5, key="ngarch_num_forecast_steps")

        if st.button("2. Lakukan Prediksi Volatilitas ▶️", key="run_ngarch_prediction_button"):
            try:
                with st.spinner("Melakukan prediksi volatilitas NGARCH... ⏳"):
                    # NGARCH forecast dari akhir data yang digunakan untuk melatih NGARCH (yaitu, residual ARIMA)
                    last_garch_obs_index = arima_residuals.index.max()

                    # Horizon adalah jumlah langkah ke depan
                    # Kita ingin memprediksi volatilitas untuk periode yang sama dengan prediksi harga masa depan
                    # plus periode data uji (jika kita ingin membandingkan)
                    total_forecast_horizon_vol = num_forecast_steps_vol # Hanya prediksi masa depan

                    # Lakukan forecast NGARCH
                    # Menggunakan 'simulation' karena terkadang 'analytic' tidak tersedia untuk semua distribusi/model
                    # Atau bisa juga menggunakan predict_in_sample() dan forecast() secara terpisah
                    # For simplicity, let's use forecast with horizon directly for out-of-sample
                    
                    # NOTE: arch_model.forecast() hanya melakukan out-of-sample forecast.
                    # Untuk mendapatkan in-sample conditional volatility untuk test set, kita gunakan conditional_volatility
                    
                    # Forecast out-of-sample volatility
                    forecast_res_ngarch_out = model_ngarch_fit.forecast(horizon=num_forecast_steps_vol,
                                                                         start=last_garch_obs_index,
                                                                         method='simulation', simulations=1000)
                    
                    # Ambil rata-rata dari simulasi varians
                    conditional_variance_forecast_mean = forecast_res_ngarch_out.variance.mean.iloc[-1, :] # Ambil baris terakhir (mean)
                    conditional_volatility_forecast = np.sqrt(conditional_variance_forecast_mean)

                    # Buat indeks untuk volatilitas yang diprediksi masa depan
                    if isinstance(last_garch_obs_index, pd.Timestamp):
                        future_vol_dates = pd.date_range(start=last_garch_obs_index + pd.Timedelta(days=1), periods=num_forecast_steps_vol, freq='D')
                    else:
                        future_vol_dates = pd.RangeIndex(start=last_garch_obs_index + 1, stop=last_garch_obs_index + 1 + num_forecast_steps_vol)

                    volatility_forecast_series = pd.Series(conditional_volatility_forecast, index=future_vol_dates)

                    st.success("Prediksi volatilitas berhasil dilakukan! 🎉")

                    # --- Visualisasi Hasil Prediksi Volatilitas ---
                    st.subheader(f"3. Prediksi Volatilitas (NGARCH) untuk {st.session_state.get('selected_currency', '')} 🌪️")
                    fig_volatility = go.Figure()
                    # Menampilkan volatilitas historis yang dimodelkan oleh NGARCH
                    fig_volatility.add_trace(go.Scatter(x=model_ngarch_fit.conditional_volatility.index, y=model_ngarch_fit.conditional_volatility, mode='lines', name='Volatilitas Kondisional (Historis)', line=dict(color='#8c564b')))
                    # Menampilkan prediksi volatilitas ke depan
                    fig_volatility.add_trace(go.Scatter(x=volatility_forecast_series.index, y=volatility_forecast_series.values, mode='lines', name='Prediksi Volatilitas (NGARCH)', line=dict(color='#9467bd', dash='dot')))
                    fig_volatility.update_layout(title_text=f'Prediksi Volatilitas NGARCH {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_volatility)

                    st.session_state['last_forecast_volatility_ngarch'] = volatility_forecast_series.iloc[-1] if not volatility_forecast_series.empty else None
                    st.session_state['volatility_forecast_series'] = volatility_forecast_series # Simpan untuk interpretasi

                    # Opsi untuk mengunduh prediksi
                    forecast_df_vol_to_save = pd.DataFrame({
                        f'Predicted_Volatility_{st.session_state.get("selected_currency", "")}': volatility_forecast_series
                    })
                    st.download_button(
                        label=f"Unduh Prediksi Volatilitas {st.session_state.get('selected_currency', '')} sebagai CSV ⬇️",
                        data=forecast_df_vol_to_save.to_csv().encode('utf-8'),
                        file_name=f'forecast_{st.session_state.get("selected_currency", "")}_ngarch_volatility.csv',
                        mime='text/csv',
                    )


            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi volatilitas: {e} ❌")
                st.info("Harap periksa kembali langkah 'MODEL NGARCH' atau ordo model yang dipilih. Pastikan model telah dilatih dan data tersedia. ⚠️")
    else:
        st.info("Harap pastikan semua langkah sebelumnya (Input Data, Data Preprocessing, Data Splitting, MODEL ARIMA (untuk residual), MODEL NGARCH) telah selesai dan mata uang telah dipilih. ⬆️")


elif st.session_state['current_page'] == 'interpretasi_saran':
    st.markdown('<div class="main-header">Interpretasi Hasil dan Saran 💡📈</div>', unsafe_allow_html=True)
    st.write(f"Bagian ini memberikan interpretasi terhadap hasil pemodelan ARIMA-NGARCH dan beberapa saran praktis untuk {st.session_state.get('selected_currency', '')}. 🤔")

    st.subheader("Interpretasi Model ARIMA 🧠📖")
    st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
    st.write("""
    Model ARIMA (AutoRegressive Integrated Moving Average) digunakan untuk memodelkan komponen mean dari data return nilai tukar.
    Ini membantu kita memprediksi arah pergerakan nilai tukar di masa depan. ⬆️⬇️
    """)
    if 'model_arima_fit' in st.session_state:
        st.write(f"- Ordo ARIMA yang digunakan adalah: {st.session_state['model_arima_fit'].model.order} 🔢")
        st.write("Parameter-parameter ini menentukan berapa banyak observasi masa lalu yang digunakan untuk memprediksi nilai saat ini (AR), seberapa banyak differencing yang dilakukan (I), dan seberapa banyak kesalahan prediksi masa lalu yang digunakan (MA).")
    st.write('</div>', unsafe_allow_html=True)

    st.subheader("Interpretasi Model NGARCH 🌪️📖")
    st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
    st.write("""
    Model NGARCH (Non-linear Generalized Autoregressive Conditional Heteroskedasticity), yang diimplementasikan di sini sebagai GJR-GARCH,
    digunakan untuk memodelkan volatilitas (varians bersyarat) dari residual model ARIMA. 🌪️
    """)
    if 'model_ngarch_fit' in st.session_state:
        st.write(f"- Ordo NGARCH (GJR-GARCH) yang digunakan: p={st.session_state['model_ngarch_fit'].model.p}, o={st.session_state['model_ngarch_fit'].model.o}, q={st.session_state['model_ngarch_fit'].model.q} 🔢")
        st.write("Model NGARCH menangkap volatilitas yang berubah seiring waktu dan juga efek asimetris (misalnya, berita buruk mungkin memiliki dampak yang lebih besar pada volatilitas dibandingkan berita baik). ⚖️")
        
        # Mengekstrak skewness dari hasil fit jika distribusi mendukung
        if st.session_state['model_ngarch_fit'].dist_info['name'] in ['skewt']:
             # arch_model.fit().skewness mungkin tidak langsung tersedia di semua versi, ambil dari params jika ada
             if 'skew' in st.session_state['model_ngarch_fit'].params.index:
                skewness_val = st.session_state['model_ngarch_fit'].params['skew']
                st.write(f"- Skewness residual untuk {st.session_state.get('selected_currency', '')} (NGARCH): {skewness_val:.4f}")
                if abs(skewness_val) > 0.1:
                    st.write("  -> Terdapat asimetri signifikan pada residual. Ini menunjukkan bahwa dampak berita baik dan buruk pada volatilitas mungkin berbeda. Model NGARCH (GJR-GARCH) secara spesifik menangani hal ini. ⚖️")
                else:
                    st.write("  -> Tidak ada asimetri signifikan pada residual. Menggunakan model yang sudah difit (bisa GARCH/NGARCH). 👍")
        elif st.session_state['model_ngarch_fit'].dist_info['name'] == 't':
            st.write(f"- Distribusi error: t-Student. Ini menangani *fat tails* (probabilitas kejadian ekstrem yang lebih tinggi) pada residual. 📊")
        else:
            st.write(f"- Distribusi error: {st.session_state['model_ngarch_fit'].dist_info['name']}. ")

    st.write('</div>', unsafe_allow_html=True)

    st.subheader("Evaluasi Kinerja Model ⭐💯")
    st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
    if 'mape_price_arima' in st.session_state:
        st.write(f"**Prediksi Nilai Tukar ({st.session_state.get('selected_currency', '')}):**")
        st.write(f"RMSE (Root Mean Squared Error): {st.session_state['rmse_price_arima']:.4f} 👇")
        st.write(f"MAE (Mean Absolute Error): {st.session_state['mae_price_arima']:.4f} 👇")
        st.write(f"MAPE (Mean Absolute Percentage Error): {st.session_state['mape_price_arima']:.2f}% 👇")
        st.write("Nilai RMSE, MAE, dan MAPE yang lebih rendah menunjukkan akurasi prediksi nilai tukar yang lebih baik. ✅")
    else:
        st.info("Silakan jalankan prediksi harga terlebih dahulu di halaman 'PREDIKSI ARIMA' untuk melihat metrik evaluasi. ➡️")
    
    st.write("""
    **Volatilitas:**
    Evaluasi akurasi prediksi volatilitas lebih kompleks karena volatilitas 'aktual' tidak dapat langsung diamati.
    Namun, model NGARCH memberikan perkiraan volatilitas bersyarat yang dapat digunakan sebagai indikator risiko.
    Semakin tinggi volatilitas yang diprediksi, semakin besar fluktuasi nilai tukar yang diantisipasi. 📈📉⚠️
    """)
    st.write('</div>', unsafe_allow_html=True)


    st.subheader("Kesimpulan dan Saran ✅💡")
    st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
    st.write("""
    Model ARIMA-NGARCH adalah alat yang kuat untuk memprediksi nilai tukar dan volatilitasnya.
    Prediksi nilai tukar membantu dalam pengambilan keputusan investasi atau transaksi di masa depan,
    sementai prediksi volatilitas memberikan wawasan tentang tingkat risiko yang mungkin terjadi.
    Ini adalah alat yang sangat berguna! 🛠️
    """)

    combined_forecast_price = None
    combined_forecast_volatility = None

    if 'future_predicted_prices_series' in st.session_state and not st.session_state['future_predicted_prices_series'].empty:
        combined_forecast_price = st.session_state['future_predicted_prices_series'].iloc[-1]
    
    if 'volatility_forecast_series' in st.session_state and not st.session_state['volatility_forecast_series'].empty:
        combined_forecast_volatility = st.session_state['volatility_forecast_series'].iloc[-1]


    if combined_forecast_price is not None and combined_forecast_volatility is not None:
        st.write(f"**Berdasarkan prediksi terbaru untuk {st.session_state.get('selected_currency', '')}:**")
        st.write(f"- Nilai tukar yang diprediksi untuk periode selanjutnya: **{combined_forecast_price:.4f}** 💰")
        st.write(f"- Prediksi volatilitas untuk periode selanjutnya: **{combined_forecast_volatility:.4f}** (semakin tinggi, semakin besar fluktuasi yang diharapkan) 🌪️")
        st.write("*(Catatan: Prediksi ini adalah berdasarkan data dan model yang dilatih, selalu perbarui model dengan data terbaru untuk hasil yang relevan.) 🔄*")

    else:
        st.info("Silakan jalankan prediksi harga di 'PREDIKSI ARIMA' dan prediksi volatilitas di 'PREDIKSI NGARCH' untuk melihat ringkasan prediksi. ➡️")

    st.write("""
    **Saran:**
    <ol>
        <li><b>Validasi Ulang Model:</b> Model time series sensitif terhadap perubahan kondisi pasar. Lakukan validasi dan pelatihan ulang model secara berkala dengan data terbaru. ♻️</li>
        <li><b>Perbandingan Model:</b> Pertimbangkan untuk membandingkan kinerja ARIMA-NGARCH dengan model lain (misalnya, GARCH murni, E-GARCH, atau model pembelajaran mesin) untuk menemukan yang paling sesuai dengan karakteristik data Anda. ⚖️</li>
        <li><b>Analisis Residual:</b> Selalu periksa residual model (terutama ARIMA) untuk memastikan tidak ada pola yang tersisa, yang mengindikasikan bahwa model belum menangkap semua informasi. 🧐</li>
        <li><b>Wawasan Domain:</b> Gabungkan hasil prediksi dengan wawasan ekonomi, geopolitik, dan sentimen pasar yang relevan untuk pengambilan keputusan yang lebih komprehensif. 💡</li>
        <li><b>Manajemen Risiko:</b> Gunakan prediksi volatilitas sebagai indikator risiko untuk menyesuaikan strategi investasi atau hedging Anda. 🛡️</li>
    </ol>
    """)
    st.write('</div>', unsafe_allow_html=True)

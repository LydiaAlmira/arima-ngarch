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
        /* Perlebar sidebar dan geser main content */
        section[data-testid="stSidebar"] {
            width: 300px !important;
        }
        section[data-testid="stSidebar"] + div {
        padding-left: 2rem;;
        }
        /* Pastikan teks panjang tombol tidak terpotong */
        .stButton>button {
            white-space: normal !important;
            word-wrap: break-word !important;
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
    "ARIMA (Model & Prediksi)": "ARIMA (Model & Prediksi)",
    "GARCH (Model & Prediksi)": "GARCH (Model & Prediksi)",
    "NGARCH (Model & Prediksi)": "NGARCH (Model & Prediksi)",
    "INTERPRETASI & SARAN 💡": "interpretasi_saran"
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
        <li><b>HOME 🏠:</b> Halaman utama yang menjelaskan tujuan dan metode sistem prediksi.</li>
        <li><b>INPUT DATA 📥:</b> Unggah data time series nilai tukar mata uang (.csv).</li>
        <li><b>DATA PREPROCESSING 🧹:</b> Bersihkan dan transformasikan data, termasuk perhitungan return.</li>
        <li><b>STASIONERITAS DATA 📊:</b> Uji stasioneritas return (ADF), dan analisis ACF & PACF.</li>
        <li><b>DATA SPLITTING ✂️:</b> Pisahkan data menjadi data latih dan uji.</li>
        <li><b>MODEL ARIMA (Mean Equation) ⚙️:</b> 
            Bangun model ARIMA pada data return <i>dan langsung prediksi nilai tukar</i>, termasuk:
            <ul>
                <li>Penentuan ordo ARIMA (p,d,q)</li>
                <li>Uji signifikansi koefisien & asumsi residual</li>
                <li>Prediksi return & rekonstruksi nilai tukar</li>
                <li>Evaluasi performa (RMSE, MAE, MAPE)</li>
            </ul>
        </li>
        <li><b>MODEL GARCH (Volatilitas) 📉:</b> 
            Bangun model GARCH pada residual ARIMA <i>dan langsung prediksi volatilitas</i>, mencakup:
            <ul>
                <li>Penentuan ordo GARCH (p,q)</li>
                <li>Uji signifikansi & normalitas residual standar</li>
                <li>Prediksi volatilitas dan visualisasi</li>
            </ul>
        </li>
        <li><b>MODEL NGARCH (Volatilitas Asimetris) 🌪️:</b> 
            Bangun model NGARCH (dengan efek leverage) pada residual ARIMA <i>dan prediksi volatilitas asimetris</i>, termasuk:
            <ul>
                <li>Ordo NGARCH (p,o,q)</li>
                <li>Evaluasi distribusi & autokorelasi residual</li>
                <li>Visualisasi prediksi volatilitas asimetris</li>
            </ul>
        </li>
        <li><b>INTERPRETASI & SARAN 💡:</b> Penjelasan hasil akhir ARIMA-GARCH/NGARCH, analisis performa model, dan rekomendasi untuk aplikasi praktis.</li>
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
    st.write("Lakukan pembersihan dan transformasi data nilai tukar.✨")

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        df_raw = st.session_state['df_currency_raw'].copy()
        st.write(f"Data nilai tukar mentah untuk {st.session_state.get('selected_currency', '')}: 📊")
        st.dataframe(df_raw.head())

        st.subheader("Pilih Kolom Data dan Transformasi 🔄")

        # 🟦 Dropdown untuk memilih kolom numerik
        numeric_cols = [col for col in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[col])]
        
        if numeric_cols:
            selected_column = st.selectbox(
                "Pilih kolom data nilai tukar yang akan diproses:",
                options=numeric_cols,
                key="selected_column"
            )
            series_data = df_raw[selected_column]

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
        
            # 🔁 Pilih jenis return
            return_type = st.radio("Pilih jenis return yang akan digunakan:", ["Log Return", "Simple Return"], key="return_type_radio")
            st.session_state['return_type'] = return_type

            # Simpan hasil preprocessing ke session_state
            st.session_state['preprocessed_data'] = series_data
            st.session_state['original_prices'] = df_raw[st.session_state['selected_column']]
            st.session_state['full_prices_series'] = series_data  # Tambahkan ini

            st.success("Preprocessing selesai! Data siap digunakan untuk uji stasioneritas. 🧪")
            st.write("Pratinjau data hasil preprocessing:")
            st.line_chart(series_data)
        
        else:
            st.warning("Tidak ditemukan kolom numerik untuk diproses. Pastikan file yang diunggah sesuai format.")
    else:
        st.warning("Silakan unggah data terlebih dahulu. 📂")
        
elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">Stasioneritas Data 📊🧪</div>', unsafe_allow_html=True)
    st.write(f"Untuk pemodelan time series, data harus stasioner. Kita akan menguji stasioneritas pada data {st.session_state.get('selected_currency', '')} dan memeriksa autokorelasi. 🔍")

    if 'preprocessed_data' in st.session_state and not st.session_state['preprocessed_data'].empty:
        series_to_test = st.session_state.get('final_series', st.session_state['preprocessed_data'])
        st.write(f"5 baris pertama data nilai tukar {st.session_state.get('selected_currency', '')} yang akan diuji:")
        st.dataframe(series_to_test.head())

        st.subheader("Uji Augmented Dickey-Fuller (ADF) 🤔")
        if st.button("Jalankan Uji ADF ▶️", key="run_adf_test"):
            try:
                result_adf = adfuller(series_to_test)
                st.session_state['adf_result'] = result_adf
                st.session_state['adf_pvalue'] = result_adf[1]
                st.write(f"**Statistik ADF:** {result_adf[0]:.4f}")
                st.write(f"**P-value:** {result_adf[1]:.4f}")
                st.write(f"**Jumlah Lags Optimal:** {result_adf[2]}")
                st.write("**Nilai Kritis:**")
                for key, value in result_adf[4].items():
                    st.write(f"  {key}: {value:.4f}")

                if result_adf[1] <= 0.05:
                    st.success("Data **stasioner** (tolak H0: ada akar unit). ✅")
                    st.session_state['is_stationary_adf'] = True
                    st.session_state['final_series'] = series_to_test
                    st.session_state['processed_returns'] = series_to_test
                else:
                    st.warning("Data **tidak stasioner** (gagal tolak H0: ada akar unit). ⚠️")
                    st.info("Akan dilakukan transformasi differencing secara otomatis... 🔄")
                    differenced = series_to_test.diff().dropna() 
                    st.session_state['differenced_data'] = differenced
                    st.write("Hasil data setelah differencing (5 baris pertama):")
                    st.dataframe(differenced.head())
                    
                    st.subheader("Uji ADF pada Data Setelah Differencing 📉")
                    result_adf_diff = adfuller(differenced)
                    st.session_state['adf_diff_result'] = result_adf_diff
                    st.session_state['adf_diff_pvalue'] = result_adf_diff[1] 
                    st.write(f"**Statistik ADF:** {result_adf_diff[0]:.4f}")
                    st.write(f"**P-value:** {result_adf_diff[1]:.4f}")
                    st.write(f"**Jumlah Lags Optimal:** {result_adf_diff[2]}")
                    st.write("**Nilai Kritis:**")
                    for key, value in result_adf_diff[4].items():
                        st.write(f"  {key}: {value:.4f}")

                    if result_adf_diff[1] <= 0.05: 
                        st.success("Setelah differencing, data menjadi **stasioner**. ✅") 
                        st.session_state['is_stationary_adf'] = True
                        st.session_state['final_series'] = differenced
                        st.session_state['processed_returns'] = differenced 
                    else:
                        st.warning("Data masih **tidak stasioner** setelah satu kali differencing. ❗")
                        st.session_state['is_stationary_adf'] = False
            except Exception as e:
                 st.error(f"Terjadi kesalahan saat menjalankan Uji ADF: {e}")

    # Plot ACF & PACF hanya jika data sudah stasioner
    if st.session_state.get('adf_pvalue', 1.0) <= 0.05 or st.session_state.get('adf_diff_pvalue', 1.0) <= 0.05:
        st.subheader("Autocorrelation Function (ACF) dan Partial Autocorrelation Function (PACF) 📈📉")
        st.info("Plot ACF menunjukkan korelasi antar lag. Plot PACF menunjukkan korelasi parsial setelah efek lag sebelumnya dihilangkan.")

        lags = st.slider("Jumlah Lags untuk Plot ACF/PACF:", 5, 50, 20, key="acf_pacf_lags")

        if st.button("Tampilkan Plot ACF dan PACF 📊", key="show_acf_pacf"):
            try:
                # Plot ACF
                fig_acf, ax_acf = plt.subplots(figsize=(8, 4))  # Lebar 8, tinggi 4 inch
                plot_acf(st.session_state['final_series'], lags=lags, alpha=0.05, ax=ax_acf)
                ax_acf.set_title(f"ACF {st.session_state.get('selected_currency', '')}")
                st.pyplot(fig_acf)
                
                # Plot PACF
                fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))  # Lebar 8, tinggi 4 inch
                plot_pacf(st.session_state['final_series'], lags=lags, alpha=0.05, ax=ax_pacf)
                ax_pacf.set_title(f"PACF {st.session_state.get('selected_currency', '')}")
                st.pyplot(fig_pacf)

                st.success("Plot ACF dan PACF berhasil ditampilkan! 🎉")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat membuat plot ACF/PACF: {e} ❌")
    else:
        st.info("Data belum stasioner. Silakan jalankan uji ADF terlebih dahulu. ⚠️")
        
elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">Data Splitting ✂️📊</div>', unsafe_allow_html=True)
    st.write(f"Pisahkan data menjadi set pelatihan dan pengujian untuk melatih dan mengevaluasi model ARIMA. Pembagian dilakukan secara berurutan karena ini adalah data time series. 📏")

    if 'processed_returns' in st.session_state and not st.session_state['processed_returns'].empty:
        data_to_split = st.session_state['processed_returns']
        currency_name = st.session_state.get('selected_currency', '')
        d_value = st.session_state.get('d', 0)

        st.write(f"Data yang telah stasioner dari {currency_name} (differencing ke-{d_value}) yang akan dibagi 📈:")

        if d_value > 1:
            st.warning(f"Data memerlukan differencing sebanyak {d_value} kali. ⚠️ Ini menunjukkan adanya non-stasioneritas kuat. Pertimbangkan untuk mengevaluasi kembali transformasi data atau model yang digunakan.")

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

            # Simpan harga asli untuk keperluan rekonstruksi prediksi
            if 'original_prices' in st.session_state:
                original_prices = st.session_state['original_prices']
                relevant_index = train_data_returns.index.union(test_data_returns.index)
                st.session_state['original_prices_for_reconstruction'] = original_prices.loc[original_prices.index.intersection(relevant_index)]

            st.success("Data berhasil dibagi! ✅")
            st.write(f"Ukuran data pelatihan: {len(train_data_returns)} sampel 💪")
            st.write(f"Ukuran data pengujian: {len(test_data_returns)} sampel 🧪")

            st.subheader(f"Visualisasi Pembagian Data {currency_name} 📈📉")
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(x=train_data_returns.index, y=train_data_returns.values, mode='lines', name='Data Pelatihan', line=dict(color='#3f72af')))
            fig_split.add_trace(go.Scatter(x=test_data_returns.index, y=test_data_returns.values, mode='lines', name='Data Pengujian', line=dict(color='#ff7f0e')))
            fig_split.update_layout(title_text=f'Pembagian Data {currency_name}', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_split)
    else:
        st.warning("Tidak ada data yang tersedia untuk dibagi. Pastikan Anda telah melalui 'Input Data', 'Preprocessing', dan 'Stasioneritas Data'. ⚠️⬆️")

elif st.session_state['current_page'] == 'ARIMA (Model & Prediksi)':
    st.markdown('<div class="main-header">MODEL & PREDIKSI ARIMA 📈</div>', unsafe_allow_html=True)
    st.write(f"Bangun dan evaluasi model ARIMA pada data return mata uang {st.session_state.get('selected_currency', '')} Gunakan juga untuk memprediksi nilai tukar. 📊")

    # Validasi data
    if 'train_data_returns' in st.session_state and not st.session_state['train_data_returns'].empty:
        train_data_returns = st.session_state['train_data_returns']
        st.write(f"Data pelatihan return ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_returns.head())

        # =========== 1. PEMODELAN ARIMA ============
        st.subheader("1. Tentukan Ordo ARIMA (p, d, q) 🔢")
        st.info("Berdasarkan plot ACF dan PACF di bagian 'Stasioneritas Data', Anda dapat memperkirakan ordo (p, q). Ordo differencing (d) harus 0 karena Anda sudah bekerja dengan data return yang diharapkan stasioner.")
        p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
        d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=0, value=0, help="Untuk data return, 'd' harus 0 karena data sudah distasionerkan.", key="arima_d")
        q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

        if st.button("Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    # Simpan ke session state 
                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.session_state['arima_residuals'] = model_arima_fit.resid.dropna()
                    st.session_state['original_prices_for_reconstruction'] = st.session_state.get('full_prices_series', None)

                    st.success("Model ARIMA berhasil dilatih! 🎉")

                    # Ringkasan 
                    st.subheader("2. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    # P-value koefisien 
                    st.subheader("3. Uji Signifikansi Koefisien (P-value) ✅❌")
                    results_table = model_arima_fit.summary().tables[1]
                    df_results = pd.read_html(model_arima_fit.summary().tables[1].as_html(), header=0, index_col=0)[0]
                    st.dataframe(df_results[['P>|z|']].style.applymap(
                        lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P ≥ 0.05)")

                    # Uji residual
                    st.subheader("4. Uji Asumsi Residual Model ARIMA 📊")
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
                        st.warning("Residual ARIMA kosong atau tidak valid untuk pengujian. ❌")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e} ❌ Pastikan data pelatihan tidak kosong dan ordo ARIMA sesuai.")
                st.info("Kesalahan umum: data terlalu pendek untuk ordo yang dipilih, atau ada nilai NaN/Inf.")
    else:
        st.warning("Data pelatihan return belum tersedia. Silakan lakukan splitting.")

    # ====== PREDIKSI ARIMA ======
    if 'model_arima_fit' in st.session_state and \
       'test_data_returns' in st.session_state and \
       'original_prices_for_reconstruction' in st.session_state:

        model_arima_fit = st.session_state['model_arima_fit']
        test_data_returns = st.session_state['test_data_returns']
        train_data_returns = st.session_state['train_data_returns']
        original_prices = st.session_state['original_prices_for_reconstruction']
        return_type = st.session_state.get('return_type', "Log Return")

        # ❗ Validasi: hentikan eksekusi jika belum tersedia
        if return_type is None:
            st.warning("Jenis return belum dipilih atau tidak tersedia di session. Silakan lakukan preprocessing ulang. ⚠️")
            st.stop()

        st.subheader("5. Prediksi Return dengan Model ARIMA 🔮")
        forecast_steps = st.slider("Langkah ke depan:", 1, len(test_data_returns), len(test_data_returns))
        start = len(train_data_returns)
        end = start + forecast_steps - 1

        try:
            # Prediksi return
            forecast_returns = model_arima_fit.predict(start=start, end=end)
            forecast_returns.index = test_data_returns.index[:forecast_steps]
            st.session_state['arima_forecast_returns'] = forecast_returns
            st.dataframe(forecast_returns.head())
            st.success("Prediksi return dengan ARIMA berhasil! 🎉")
         
            st.subheader("6. Rekonstruksi Prediksi Nilai Tukar dari Return 🔄")
            
            # Ambil harga terakhir dari training untuk rekonstruksi
            last_train_price = original_prices.loc[train_data_returns.index[-1]]
            arima_forecast_returns = forecast_returns  # Pastikan variabel ini didefinisikan
            reconstructed_prices = pd.Series(index=forecast_returns.index, dtype=float)
            previous_price = last_train_price

            for date, forecast_return in forecast_returns.items():
                if return_type == "Log Return":
                    current_price = previous_price * np.exp(forecast_return)
                else: 
                    current_price = previous_price * (1 + forecast_return)
                reconstructed_prices.loc[date] = current_price
                previous_price = current_price 

            st.session_state['arima_forecast_prices'] = reconstructed_prices
            st.success("Prediksi nilai tukar berhasil direkonstruksi! 🎉")
            st.write("5 nilai prediksi nilai tukar pertama:")
            st.dataframe(reconstructed_prices.head())

            st.subheader("7. Visualisasi Prediksi Nilai Tukar ARIMA vs. Aktual 📊")
            fig_arima_forecast = go.Figure()
            
            # Plot data historis (latih + uji harga asli)
            fig_arima_forecast.add_trace(go.Scatter(
                x=original_prices.index,
                y=original_prices.values,
                mode='lines',
                name=f'Nilai Tukar Asli ({st.session_state.get("selected_currency", "")})',
                line=dict(color='#1f77b4')
            ))

            # Plot prediksi ARIMA
            fig_arima_forecast.add_trace(go.Scatter(
                x=reconstructed_prices.index,
                y=reconstructed_prices.values,
                mode='lines',
                name='Prediksi ARIMA (Nilai Tukar)',
                line=dict(color='#ff7f0e', dash='dash')
            ))

            fig_arima_forecast.update_layout(
                title=f'Prediksi Nilai Tukar {st.session_state.get("selected_currency", "")} dengan ARIMA vs. Data Aktual',
                xaxis_title='Tanggal',
                yaxis_title='Nilai Tukar',
                xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig_arima_forecast)

            st.subheader("8. Evaluasi Model ARIMA 🧪")
            st.info("Metrik evaluasi seperti RMSE, MAE, dan MAPE digunakan untuk mengukur akurasi prediksi.")

            # Menyelaraskan indeks untuk evaluasi
            actual_prices_test = original_prices.loc[test_data_returns.index[0]:test_data_returns.index[-1]]
            actual_prices_test = actual_prices_test.iloc[:forecast_steps]

            # Pastikan indeks prediksi dan aktual sama persis
            common_index = actual_prices_test.index.intersection(reconstructed_prices.index)
            actual_prices_test_aligned = actual_prices_test.loc[common_index]
            predicted_prices_aligned = reconstructed_prices.loc[common_index]

            if not actual_prices_test_aligned.empty:
                rmse_arima = np.sqrt(np.mean((predicted_prices_aligned - actual_prices_test_aligned)**2))
                mae_arima = np.mean(np.abs(predicted_prices_aligned - actual_prices_test_aligned))
                mape_arima = np.mean(np.abs((actual_prices_test_aligned - predicted_prices_aligned) / actual_prices_test_aligned.replace(0, np.nan))) * 100

                st.write(f"**RMSE (Root Mean Squared Error):** {rmse_arima:.4f}")
                st.write(f"**MAE (Mean Absolute Error):** {mae_arima:.4f}")
                st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape_arima:.2f}%")
            else:
                st.warning("Tidak ada data aktual yang cocok untuk evaluasi. Pastikan rentang indeks data uji sesuai dengan prediksi. 🤷")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi atau merekonstruksi nilai tukar dengan ARIMA: {e} ❌")
            st.info("Pastikan model ARIMA sudah dilatih dengan benar dan data test tersedia.")
    else:
        st.info("Silakan latih model ARIMA di halaman 'Model ARIMA' dan pastikan data splitting sudah dilakukan. 📈✂️")

elif st.session_state['current_page'] == 'GARCH (Model & Prediksi)':
    st.markdown('<div class="main-header">GARCH (Model & Prediksi) 🌪️📈</div>', unsafe_allow_html=True)
    st.write(f"Bangun dan evaluasi model GARCH untuk memodelkan volatilitas dari residual ARIMA pada mata uang {st.session_state.get('selected_currency', '')}. Juga prediksi volatilitas ke depan.")

    if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
        arima_residuals = st.session_state['arima_residuals']
        st.write("Data residual ARIMA yang digunakan:")
        st.dataframe(arima_residuals.head())

        st.subheader("1. Tentukan Ordo GARCH (p, q) 🔢")
        garch_p = st.number_input("ARCH Order (p):", min_value=1, max_value=5, value=1, key="garch_p")
        garch_q = st.number_input("GARCH Order (q):", min_value=1, max_value=5, value=1, key="garch_q")

        if st.button("Latih Model GARCH ▶️", key="train_garch_button"):
            try:
                from arch import arch_model
                returns_for_garch = arima_residuals.dropna()
                
                with st.spinner("Melatih model GARCH..."):
                    garch_model = arch_model(
                        returns_for_garch,
                        mean="zero",
                        vol="Garch",
                        p=garch_p,
                        q=garch_q,
                        dist="t"
                    )
                    model_garch_fit = garch_model.fit(disp="off")
                    st.session_state["model_garch_fit"] = model_garch_fit
                    st.success("Model GARCH berhasil dilatih! 🎉")

                    # Ringkasan
                    st.subheader("2. Ringkasan Model GARCH (Koefisien dan Statistik) 📝")
                    st.text(model_garch_fit.summary().as_text())

                    # Uji Signifikansi
                    st.subheader("3. Uji Signifikansi Koefisien GARCH ✅❌")
                    df_coef = pd.DataFrame({
                        "Koefisien": model_garch_fit.params,
                        "t-Stat": model_garch_fit.tvalues,
                        "P-Value": model_garch_fit.pvalues
                    })
                    st.dataframe(df_coef.style.applymap(
                       lambda x: 'background-color: #d4edda' if isinstance(x, float) and x < 0.05 else 'background-color: #f8d7da',
                       subset=["P-Value"]
                    ))
                    st.caption("Hijau: signifikan (P < 0.05), Merah: tidak signifikan (P ≥ 0.05)")

                    # Uji Residual
                    st.subheader("4. Uji Residual Standar GARCH 📊")
                    std_resid = model_garch_fit.resid / model_garch_fit.conditional_volatility
                    st.session_state["garch_std_residuals"] = std_resid

                    st.write("##### Plot Residual Standar")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=std_resid.index, y=std_resid, mode="lines", name="Std Residual", line=dict(color="green")))
                    fig.update_layout(title="Residual Standar GARCH", xaxis_title="Tanggal", yaxis_title="Nilai")
                    st.plotly_chart(fig)

                    # Uji Normalitas
                    st.write("##### Uji Normalitas (Jarque-Bera)")
                    jb_stat, jb_p = stats.jarque_bera(std_resid.dropna())
                    st.write(f"Statistik JB: {jb_stat:.4f}, P-value: {jb_p:.4f}")
                    if jb_p > 0.05:
                        st.success("Residual standar **terdistribusi normal**. ✅")
                    else:
                        st.warning("Residual standar **tidak normal** (tolak H0). ⚠️")

                    # Ljung-Box (Autokorelasi)
                    st.write("##### Uji Autokorelasi (Ljung-Box)")
                    lb = sm.stats.acorr_ljungbox(std_resid.dropna(), lags=[10], return_df=True)
                    st.write(lb)
                    if lb["lb_pvalue"].iloc[0] > 0.05:
                        st.success("Tidak ada autokorelasi signifikan. ✅")
                    else:
                        st.warning("Terdapat autokorelasi signifikan. ⚠️")

                    # ARCH Effect 
                    st.write("##### Uji ARCH Effect (Residual Kuadrat)")
                    lb_sq = sm.stats.acorr_ljungbox(std_resid.dropna()**2, lags=[10], return_df=True)
                    st.write(lb_sq)
                    if lb_sq["lb_pvalue"].iloc[0] > 0.05:
                        st.success("ARCH effect berhasil ditangkap oleh model GARCH. ✅")
                    else:
                        st.warning("Model GARCH mungkin belum cukup menangkap ARCH effect. ⚠️")

                # Prediksi Volatilitas ke depan
                st.subheader("5. Prediksi Volatilitas ke Depan 🔮")
                forecast_horizon = st.slider("Jumlah hari ke depan:", 1, 30, 5, key="forecast_garch_horizon")
                garch_forecast = model_garch_fit.forecast(horizon=forecast_horizon)
                forecast_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
                dates = pd.date_range(start=std_resid.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='B')
                forecast_vol_series = pd.Series(data=forecast_volatility, index=dates)
                st.line_chart(forecast_vol_series)
                st.session_state['garch_forecast_volatility'] = forecast_vol_series
                st.write("5 prediksi volatilitas pertama:")
                st.dataframe(forecast_vol_series.head())

            except Exception as e:
                st.error(f"Terjadi kesalahan saat pelatihan GARCH: {e}")
        
elif st.session_state['current_page'] == 'NGARCH (Model & Prediksi)':
    st.markdown('<div class="main-header">MODEL & PREDIKSI NGARCH 🌪️</div>', unsafe_allow_html=True)
    st.write("Modelkan dan prediksi volatilitas bersyarat dengan NGARCH untuk menangkap efek asimetri pada volatilitas. 📊")

    # Pastikan residual ARIMA tersedia
    if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
        residuals = st.session_state['arima_residuals'].dropna()
        st.write("Data residual ARIMA yang digunakan:")
        st.dataframe(residuals.head())

        st.subheader("1. Tentukan Ordo NGARCH (p, q) 🔢")
        st.info("Untuk NGARCH(p, q), 'p' adalah ordo ARCH (jumlah lag dari residual kuadrat) dan 'q' adalah ordo GARCH (jumlah lag dari varians bersyarat). Umumnya GARCH(1,1) adalah titik awal yang baik.")
        ngarch_p = st.number_input("Ordo ARCH (p):", min_value=1, max_value=5, value=1, key="ngarch_p")
        ngarch_o = st.number_input("Ordo Asymmetric (o):", min_value=0, max_value=1, value=1, help="Ordo asimetris untuk efek leverage. Set ke 0 untuk GARCH biasa. Set ke 1 untuk NGARCH/GJR-GARCH.", key="ngarch_o")
        ngarch_q = st.number_input("Ordo GARCH (q):", min_value=1, max_value=5, value=1, key="ngarch_q")

        if st.button("Latih Model NGARCH ▶️", key="train_ngarch_button"):

            try:
                from arch.univariate import GARCH, ConstantMean
                from arch.__future__ import reindexing

                with st.spinner("Melatih model NGARCH..."):
                    # Ambil residual dari ARIMA
                    returns_for_ngarch = st.session_state.get("arima_residuals", None)
                    if returns_for_ngarch is None or returns_for_ngarch.isna().all():
                        st.error("Residual ARIMA tidak tersedia. Latih model ARIMA terlebih dahulu.")
                        st.stop()
                        
                    # Dapatkan nilai parameter dari session_state (atau gunakan default jika belum ada)
                    p_ngarch = st.session_state.get('p_ngarch', 1)
                    q_ngarch = st.session_state.get('q_ngarch', 1)
                    o_ngarch = st.session_state.get('o_ngarch', 1)

                    # Buat dan latih model
                    ngarch_model = arch_model(
                        returns_for_ngarch,
                        mean='zero',
                        vol='Garch',
                        p=p_ngarch,
                        o=o_ngarch,
                        q=q_ngarch,
                        dist='t'
                    )
                    ngarch_fit = ngarch_model.fit(disp='off')
                    st.session_state['model_ngarch_fit'] = ngarch_fit
                    st.success("Model NGARCH berhasil dilatih! 🎉")
            
                    st.subheader("2. Ringkasan Model NGARCH")
                    st.text(ngarch_fit.summary().as_text())

                    st.subheader("3. Uji Signifikansi Koefisien NGARCH ✅❌")
                    params = ngarch_fit.params
                    pvals = ngarch_fit.pvalues
                    df_coef = pd.DataFrame({'Koefisien': params, 'P-Value': pvals})
                    st.dataframe(df_coef.style.applymap(
                        lambda x: 'background-color: #d4edda' if isinstance(x, float) and x < 0.05 else 'background-color: #f8d7da',
                        subset=['P-Value']
                    ))

                    # Residual standar
                    std_resid = ngarch_fit.std_resid.dropna()
                    st.session_state['ngarch_std_residuals'] = std_resid
                
                    st.subheader("4. Evaluasi Residual Standar NGARCH 📊")
                    std_residuals = ngarch_fit.resid / ngarch_fit.conditional_volatility
                    st.session_state['ngarch_std_residuals'] = std_residuals # Simpan residual standar

                    if not std_residuals.empty:
                        # Plot Residual Standar
                        st.write("##### Plot Residual Standar NGARCH")
                        fig_std_res = go.Figure()
                        fig_std_res.add_trace(go.Scatter(x=std_residuals.index, y=std_residuals, mode='lines', name='Residual Standar NGARCH', line=dict(color='#2ca02c')))
                        fig_std_res.update_layout(title_text=f'Residual Standar Model NGARCH ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_std_res)

                        # Uji Normalitas (Jarque-Bera) pada Residual Standar
                        st.write("##### Uji Normalitas (Jarque-Bera Test) pada Residual Standar")
                        jb_test_ngarch = stats.jarque_bera(std_residuals.dropna()) # Pastikan tidak ada NaN
                        st.write(f"Statistik Jarque-Bera: {jb_test_ngarch[0]:.4f}")
                        st.write(f"P-value: {jb_test_ngarch[1]:.4f}")
                        if jb_test_ngarch[1] > 0.05:
                            st.success("Residual standar **terdistribusi normal** (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual standar **tidak terdistribusi normal** (tolak H0). ⚠️ Ini masih umum untuk model GARCH dengan distribusi Student's t, asalkan model sudah menangkap volatilitas berkelompok.")
                            st.info("Jika Anda menggunakan distribusi Student's t atau skew-t, hasil uji normalitas mungkin masih menolak H0, tetapi ini diharapkan karena model dirancang untuk menangani *fat tails*.")

                        # Uji Autokorelasi (Ljung-Box Test) pada Residual Standar
                        st.write("##### Uji Autokorelasi (Ljung-Box Test) pada Residual Standar")
                        lb_test_ngarch = sm.stats.acorr_ljungbox(std_residuals.dropna(), lags=[10], return_df=True) # Pastikan tidak ada NaN
                        st.write(lb_test_ngarch)
                        if lb_test_ngarch['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual standar **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅")
                        else:
                            st.warning("Residual standar **memiliki autokorelasi** signifikan (tolak H0). ⚠️ Ini menunjukkan model NGARCH mungkin belum sepenuhnya menangkap dependensi.")
                            st.info("Jika ada autokorelasi, pertimbangkan ordo NGARCH yang berbeda atau model GARCH yang lebih kompleks.")

                        # Uji Autokorelasi (Ljung-Box Test) pada Residual Standar Kuadrat
                        st.write("##### Uji Autokorelasi (Ljung-Box Test) pada Residual Standar Kuadrat")
                        lb_arch_test_ngarch = sm.stats.acorr_ljungbox(std_residuals.dropna()**2, lags=[10], return_df=True) # Pastikan tidak ada NaN
                        st.write(lb_arch_test_ngarch)
                        if lb_arch_test_ngarch['lb_pvalue'].iloc[0] > 0.05:
                            st.success("Residual standar kuadrat **tidak memiliki autokorelasi** signifikan (gagal tolak H0). ✅ Ini menunjukkan model NGARCH telah berhasil menangkap efek ARCH/GARCH.")
                        else:
                            st.warning("Residual standar kuadrat **memiliki autokorelasi** signifikan (tolak H0). ⚠️ Ini menunjukkan model NGARCH mungkin belum sepenuhnya menangkap volatilitas berkelompok.")
                            st.info("Jika ada autokorelasi pada residual kuadrat, pertimbangkan ordo NGARCH yang lebih tinggi atau model GARCH yang berbeda (misalnya, EGARCH).")
                    else:
                        st.warning("Residual standar NGARCH kosong atau tidak valid untuk pengujian. ❌")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e} ❌ Pastikan residual ARIMA tidak kosong dan ordo NGARCH sesuai.")
                st.info("Kesalahan umum: data terlalu pendek, atau ada nilai tak terhingga/NaN setelah normalisasi.")
  
    # Prediksi NGARCH
    if 'model_ngarch_fit' in st.session_state and 'test_data_returns' in st.session_state:
        st.subheader("5. Prediksi Volatilitas Bersyarat (NGARCH) 🔮")
        ngarch_fit = st.session_state['model_ngarch_fit']
        test_returns = st.session_state['test_data_returns']
        horizon = len(test_returns)

        try:
            forecast = ngarch_fit.forecast(horizon=horizon, reindex=False)
            predicted_vol = np.sqrt(forecast.variance.values[-1, :horizon])
            predicted_vol_series = pd.Series(predicted_vol, index=test_returns.index)
            st.session_state['ngarch_forecast_volatility'] = predicted_vol_series

            st.success("Prediksi volatilitas dengan NGARCH berhasil! 🎉")
            st.write("Prediksi 5 hari pertama:")
            st.dataframe(predicted_vol_series.head())

            st.subheader("6. Visualisasi Prediksi Volatilitas Bersyarat NGARCH 📊")
            fig_ngarch_forecast = go.Figure()

            # Plot volatilitas bersyarat yang dihasilkan oleh model pada data pelatihan
            # Ini adalah estimasi volatilitas historis berdasarkan model
            conditional_vol_train = ngarch_fit.conditional_volatility
            fig_ngarch_forecast.add_trace(go.Scatter(
                x=conditional_vol_train.index,
                y=conditional_vol_train.values,
                mode='lines',
                name='Volatilitas Bersyarat (In-Sample)',
                line=dict(color='#2ca02c')
            ))
            
            # Plot prediksi volatilitas
            fig_ngarch_forecast.add_trace(go.Scatter(
            x=predicted_vol_series.index,
            y=predicted_vol_series.values,
                mode='lines',
                name='Prediksi Volatilitas (Out-of-Sample)',
                line=dict(color='#d62728', dash='dash')
            ))

            fig_ngarch_forecast.update_layout(
                title=f'Prediksi Volatilitas Bersyarat {st.session_state.get("selected_currency", "")} dengan NGARCH',
                xaxis_title='Tanggal',
                yaxis_title='Volatilitas',
                xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig_ngarch_forecast)
            
            # Optionally, show actual squared returns as a proxy for actual volatility
            st.subheader("7. Perbandingan dengan Volatilitas Aktual (Squared Returns) 📉")
            st.info("Volatilitas aktual tidak dapat diamati secara langsung, tetapi kuadrat dari return adalah proksi yang umum digunakan untuk memvisualisasikan volatilitas historis.")
            
            fig_actual_vs_pred_vol = go.Figure()
            
            # Squared returns for the whole series (train + test)
            actual_squared_returns = (st.session_state['processed_returns']**2)
            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=actual_squared_returns.index,
                y=actual_squared_returns.values,
                mode='lines', 
                name='Kuadrat Return Aktual',
                line=dict(color='#8c564b'),
                opacity=0.7  # ✅ letakkan di luar line
            ))

            # Conditional volatility (in-sample)
            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=conditional_vol_train.index,
                y=conditional_vol_train.values**2, # Square it for comparison with squared returns
                mode='lines',
                name='Varians Bersyarat (In-Sample)',
                line=dict(color='#2ca02c', width=2)
            ))

            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=predicted_vol_series.index,
                y=predicted_vol_series.values**2,
                mode='lines',
                name='Prediksi Varians (Out-of-Sample)',
                line=dict(color='#d62728', dash='dash', width=2)
            ))
            
            fig_actual_vs_pred_vol.update_layout(
                title=f'Prediksi Varians NGARCH vs. Kuadrat Return Aktual {st.session_state.get("selected_currency", "")}',
                xaxis_title='Tanggal',
                yaxis_title='Varians / Kuadrat Return',
                xaxis_rangeslider_visible=True
            )
            st.plotly_chart(fig_actual_vs_pred_vol)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi volatilitas dengan NGARCH: {e} ❌")
            st.info("Pastikan model NGARCH sudah dilatih dengan benar.")
    else:
        st.info("Silakan latih model NGARCH di halaman 'Model NGARCH' terlebih dahulu. 🌪️")

elif st.session_state['current_page'] == 'interpretasi_saran':
    st.markdown('<div class="main-header">INTERPRETASI & SARAN 💡</div>', unsafe_allow_html=True)
    st.write("Di bagian ini, Anda akan menemukan interpretasi dari hasil pemodelan ARIMA-NGARCH dan saran umum untuk langkah selanjutnya. 🧠")

    st.subheader("Ringkasan Hasil Penting 📑")

    # ARIMA Interpretation
    st.markdown("#### Hasil Model ARIMA (Prediksi Nilai Tukar) 📈")
    if 'model_arima_fit' in st.session_state:
        arima_fit = st.session_state['model_arima_fit']
        arima_order = arima_fit.model_orders
        st.markdown(f"Model ARIMA({arima_order.get('ar', '-')},0,{arima_order.get('ma', '-')}) telah dilatih pada data return {st.session_state.get('selected_currency', '')}.")
        st.markdown("Berikut adalah beberapa poin penting dari ringkasan model:")
        st.markdown(f"- **AIC**: {arima_fit.aic:.2f}")
        st.markdown(f"- **BIC**: {arima_fit.bic:.2f}")
        st.info("Nilai AIC dan BIC digunakan untuk membandingkan performa antar model. Nilai lebih rendah menunjukkan model yang lebih baik secara relatif.")

        
        # Check for significant coefficients
        results_table = arima_fit.summary().tables[1]
        df_results = pd.read_html(results_table.as_html(), header=0, index_col=0)[0]
        
        significant_params = df_results[df_results['P>|z|'] < 0.05]
        insignificant_params = df_results[df_results['P>|z|'] >= 0.05]

        if not significant_params.empty:
            st.success("✅ **Koefisien Signifikan:**")
            for index, row in significant_params.iterrows():
                st.write(f"- Parameter `{index}` (P-value: {row['P>|z|']:.4f}) signifikan secara statistik, menunjukkan pengaruh yang relevan pada return nilai tukar.")
        else:
            st.warning("⚠️ **Tidak Ada Koefisien Signifikan:** Tidak ada koefisien AR atau MA yang signifikan pada tingkat 5%. Ini mungkin menunjukkan bahwa model ARIMA tidak sepenuhnya menangkap pola linier dalam data return atau ordo model perlu disesuaikan.")

        # Check Ljung-Box for ARIMA residuals
        if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
            arima_residuals = st.session_state['arima_residuals']
            lb_test = sm.stats.acorr_ljungbox(arima_residuals, lags=[10], return_df=True)
            if lb_test['lb_pvalue'].iloc[0] > 0.05:
                st.success("✅ **Residual ARIMA:** Tidak menunjukkan autokorelasi signifikan, yang berarti model ARIMA telah menangkap sebagian besar dependensi linier dalam data return.")
            else:
                st.warning("⚠️ **Residual ARIMA:** Masih menunjukkan autokorelasi signifikan. Ini mengindikasikan bahwa model ARIMA mungkin perlu disempurnakan (misalnya, dengan mengubah ordo p atau q) untuk menangkap lebih banyak pola dalam data.")
        else:
            st.info("Informasi residual ARIMA tidak tersedia.")

        # Check ARCH effect
        if 'arima_residual_has_arch_effect' in st.session_state:
            if st.session_state['arima_residual_has_arch_effect']:
                st.info("💡 **Efek ARCH:** Residual ARIMA menunjukkan efek ARCH/GARCH yang signifikan, yang membenarkan penggunaan model NGARCH untuk memodelkan volatilitas.")
            else:
                st.info("ℹ️ **Tidak Ada Efek ARCH:** Residual ARIMA tidak menunjukkan efek ARCH/GARCH yang signifikan. Meskipun model NGARCH tetap dilatih, dampaknya mungkin tidak sebesar jika efek ARCH/GARCH terdeteksi kuat.")
        
        # Prediction evaluation
        if 'arima_forecast_prices' in st.session_state and 'original_prices_for_reconstruction' in st.session_state and 'test_data_returns' in st.session_state:
            actual_prices_test = st.session_state['original_prices_for_reconstruction'].loc[st.session_state['test_data_returns'].index[0]:st.session_state['test_data_returns'].index[-1]]
            predicted_prices_aligned = st.session_state['arima_forecast_prices'].loc[actual_prices_test.index.intersection(st.session_state['arima_forecast_prices'].index)]
            actual_prices_test_aligned = actual_prices_test.loc[actual_prices_test.index.intersection(st.session_state['arima_forecast_prices'].index)]

            if not actual_prices_test_aligned.empty:
                rmse_arima = np.sqrt(np.mean((predicted_prices_aligned - actual_prices_test_aligned)**2))
                mae_arima = np.mean(np.abs(predicted_prices_aligned - actual_prices_test_aligned))
                mape_arima = np.mean(np.abs((actual_prices_test_aligned - predicted_prices_aligned) / actual_prices_test_aligned.replace(0, np.nan))) * 100
                st.markdown(f"**Evaluasi Prediksi ARIMA:**")
                st.write(f"- RMSE: {rmse_arima:.4f}")
                st.write(f"- MAE: {mae_arima:.4f}")
                st.write(f"- MAPE: {mape_arima:.2f}%")
                st.info("Nilai RMSE, MAE, dan MAPE yang lebih rendah menunjukkan akurasi prediksi yang lebih baik.")
            else:
                st.warning("Evaluasi prediksi ARIMA tidak tersedia karena data aktual dan prediksi tidak selaras.")
        else:
            st.info("Prediksi ARIMA belum tersedia untuk evaluasi.")

    else:
        st.info("Model ARIMA belum dilatih. Silakan kunjungi halaman 'Model ARIMA'.")

    # NGARCH Interpretation
    st.markdown("#### Hasil Model NGARCH (Prediksi Volatilitas) 🌪️")
    if 'model_ngarch_fit' in st.session_state:
        ngarch_fit = st.session_state['model_ngarch_fit']
        
        # Ambil p, o, q dari session_state hanya jika sudah tersedia
        p_ngarch = st.session_state.get('p_ngarch', 1)
        o_ngarch = st.session_state.get('o_ngarch', 1)
        q_ngarch = st.session_state.get('q_ngarch', 1)

        st.markdown(f"Model NGARCH({p_ngarch},{o_ngarch},{q_ngarch}) dengan distribusi residual Student's t telah dilatih pada residual ARIMA.")
        st.markdown("Berikut adalah beberapa poin penting dari ringkasan model:")
        st.markdown(f"- **AIC**: {ngarch_fit.aic:.2f}")
        st.info("Nilai AIC dan BIC digunakan untuk membandingkan performa antar model. Nilai lebih rendah menunjukkan model yang lebih baik secara relatif.")

        # Parsing hasil ringkasan model
        try:
            results_html_ngarch = ngarch_fit.summary().as_html()
            df_ngarch_results = pd.read_html(results_html_ngarch, header=0, index_col=0)[0]

            if 'P-value' in df_ngarch_results.columns:
                significant_params_ngarch = df_ngarch_results[df_ngarch_results['P-value'] < 0.05]
                insignificant_params_ngarch = df_ngarch_results[df_ngarch_results['P-value'] >= 0.05]

                if not significant_params_ngarch.empty:
                    st.success("✅ **Koefisien Signifikan:**")
                    for index, row in significant_params_ngarch.iterrows():
                        st.write(f"- Parameter `{index}` (P-value: {row['P-value']:.4f}) signifikan secara statistik.")
                        if index.startswith('alpha'):
                            st.write("  - Menunjukkan adanya efek ARCH (kejutan masa lalu berpengaruh pada volatilitas saat ini).")
                        elif index.startswith('gamma'):
                            st.write("  - Menunjukkan adanya efek leverage (berita buruk memiliki dampak lebih besar terhadap volatilitas).")
                        elif index.startswith('beta'):
                            st.write("  - Menunjukkan adanya efek GARCH (volatilitas masa lalu berpengaruh pada volatilitas saat ini).")
                        elif index == 'omega':
                            st.write("  - Merupakan konstanta dalam persamaan varians.")
                        elif index == 'nu':
                            st.write("  - Derajat kebebasan Student's t (menggambarkan ketebalan ekor distribusi).")
                else:
                    st.warning("⚠️ Tidak ada koefisien signifikan pada tingkat 5%. Mungkin model NGARCH tidak terlalu diperlukan atau perlu dikalibrasi ulang.")
            else:
                st.warning("❌ Kolom `P-value` tidak ditemukan. Struktur tabel hasil model mungkin berbeda.")

        except Exception as e:
            st.error(f"Gagal memproses ringkasan model NGARCH: {e}")

        # Uji autokorelasi pada residual standar kuadrat
        if 'ngarch_std_residuals' in st.session_state and not st.session_state['ngarch_std_residuals'].empty:
            std_residuals = st.session_state['ngarch_std_residuals']
            lb_arch_test_ngarch = sm.stats.acorr_ljungbox(std_residuals.dropna()**2, lags=[10], return_df=True)
            if lb_arch_test_ngarch['lb_pvalue'].iloc[0] > 0.05:
                st.success("✅ **Residual Standar Kuadrat NGARCH:** Tidak menunjukkan autokorelasi signifikan. Model NGARCH telah menangkap volatilitas berkelompok.")
            else:
                st.warning("⚠️ **Residual Standar Kuadrat NGARCH:** Masih menunjukkan autokorelasi. Pertimbangkan peningkatan model (misalnya EGARCH atau ordo lebih tinggi).")
        else:
            st.info("Residual standar NGARCH belum tersedia atau kosong.")

    else:
        st.info("Model NGARCH belum dilatih. Silakan latih terlebih dahulu di halaman 'NGARCH (Model & Prediksi)'.")

    st.subheader("Saran Umum dan Langkah Selanjutnya 🗺️")
    st.markdown("""
    <div class="guidance-list">
    <ul>
        <li><b>Validasi Model:</b> Selalu validasi model dengan data baru jika memungkinkan. Performa model dapat berubah seiring waktu.</li>
        <li><b>Optimasi Ordo:</b> Jika hasil uji asumsi residual menunjukkan masalah (misalnya, autokorelasi tersisa), pertimbangkan untuk mencoba kombinasi ordo (p, q) yang berbeda untuk ARIMA dan NGARCH. Kriteria informasi seperti AIC atau BIC (terlihat di ringkasan model) dapat membantu dalam pemilihan model: nilai yang lebih rendah umumnya lebih baik.</li>
        <li><b>Distribusi Residual:</b> Untuk model GARCH, eksperimen dengan distribusi residual yang berbeda (misalnya, Student's t atau Generalized Error Distribution) jika uji normalitas residual standar masih ditolak.</li>
        <li><b>Model Alternatif:</b> Jika ARIMA-NGARCH tidak memberikan hasil yang memuaskan, pertimbangkan model alternatif:
            <ul>
                <li>Untuk mean: ARMA, SARIMA (jika ada musiman).</li>
                <li>Untuk volatilitas: GARCH, EGARCH (Exponential GARCH, juga menangani efek leverage), APARCH, dll.</li>
            </ul>
        </li>
        <li><b>Faktor Eksternal:</b> Dalam skenario nyata, volatilitas mata uang sering dipengaruhi oleh berita ekonomi, kebijakan bank sentral, atau peristiwa geopolitik. Model yang lebih canggih mungkin memerlukan variabel eksogen.</li>
        <li><b>Interval Prediksi:</b> Model GARCH memungkinkan perhitungan interval prediksi yang akurat yang mencerminkan volatilitas yang diharapkan, memberikan gambaran yang lebih lengkap tentang ketidakpastian.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

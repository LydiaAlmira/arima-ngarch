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
from scipy.stats import kstest
from statsmodels.stats.diagnostic import acorr_ljungbox
import pickle
import os
from datetime import datetime

def load_data(file_source=None, default_filename=None):
    import pandas as pd

    try:
        if file_source == 'default':
            df = pd.read_csv(default_filename, sep=';', decimal='.')
        else:
            df = pd.read_csv(file_source, sep=';', decimal='.')

        # Konversi kolom angka ke numerik
        for col in df.columns:
            if col != 'Date':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Set kolom 'Date' sebagai index kalau ada
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.set_index('Date')

        return df

    except Exception as e:
        print(f"Gagal membaca data: {e}")
        return pd.DataFrame()
        
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
    "ARIMA Model": "ARIMA Model",
    "GARCH (Model & Prediksi)": "GARCH (Model & Prediksi)",
    "NGARCH (Model & Prediksi)": "NGARCH (Model & Prediksi)",
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
        <li><b>MODEL ARIMA (Mean Equation) ⚙️:</b> 
            Bangun model ARIMA pada data return <i>dan langsung prediksi nilai tukar</i>, termasuk:
            <ul>
                <li>Penentuan ordo ARIMA (p,d,q)</li>
                <li>Uji signifikansi koefisien & asumsi residual</li>
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
    st.write("Unggah data time series nilai tukar mata uang. File CSV harus memiliki kolom 'Date' dan satu atau lebih kolom mata uang. 🗂️")

    col1, col2 = st.columns(2)

    with col1:
        st.session_state['variable_name'] = st.text_input("Nama Variabel:", value=st.session_state['variable_name'], key="variable_name_input")

    df_general = pd.DataFrame()
    uploaded_file = st.file_uploader("Pilih file CSV nilai tukar Anda ⬆️", type="csv", key="input_data_uploader")

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=';')
            df.columns = df.columns.str.strip()
            
            if 'Date' not in df.columns:
                st.error("Kolom 'Date' tidak ditemukan.")
                st.stop()
           
            # Format sesuai formatmu: '01/08/2019 00:00'
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y %H:%M', errors='coerce')
            df = df.dropna(subset=['Date']).sort_values('Date')  # sort agar urut
            df.set_index('Date', inplace=True)
            
            # Pilih kolom harga
            harga_col = st.selectbox("Pilih kolom harga:", [col for col in df.columns if col != 'Date'])

            # Konversi harga Eropa ke float
            df[harga_col] = df[harga_col].astype(str) \
                .str.replace('.', '', regex=False) \
                .str.replace(',', '.', regex=False) \
                .str.replace('[^0-9.-]', '', regex=True)

            df[harga_col] = pd.to_numeric(df[harga_col], errors='coerce')
            df = df.dropna(subset=[harga_col])

            # Simpan ke session
            df = df.sort_index()
            st.session_state['df_currency_raw'] = df[[harga_col]].rename(columns={harga_col: 'Value'})
            st.session_state['df_currency_raw_multi'] = df
            st.session_state['selected_currency'] = harga_col
            st.session_state['variable_name'] = harga_col

            # Tampilkan hasil
            st.dataframe(df.head())
            st.success(f"Data berhasil dimuat. Periode: {df.index.min().date()} sampai {df.index.max().date()}")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")
            st.stop()

    elif 'df_currency_raw_multi' not in st.session_state or st.session_state['df_currency_raw_multi'].empty:
        st.info("Tidak ada file diunggah. Anda dapat mengunggah file sendiri, atau coba muat data contoh.")
        if st.checkbox("Muat data contoh dari repositori 📂", key="load_default_checkbox"):
            try:
                df_general = pd.read_csv('data/default_currency_multi.csv', sep=';', thousands='.')
                if 'Date' in df_general.columns:
                    df_general['Date'] = pd.to_datetime(df_general['Date'], errors='coerce')
                    df_general = df_general.set_index('Date')

                for col in df_general.columns:
                    df_general[col] = pd.to_numeric(df_general[col], errors='coerce')

                st.success("Data contoh berhasil dimuat.")
                st.dataframe(df_general.head())
            except Exception as e:
                st.error(f"Gagal memuat data contoh: {e}")
                st.stop()
        else:
            st.warning("Silakan unggah file CSV Anda untuk memulai.")
            st.session_state['df_currency_raw_multi'] = pd.DataFrame()
            st.session_state['df_currency_raw'] = pd.DataFrame()
            st.stop()
    else:
        st.write("✅ Menggunakan data nilai tukar yang dimuat sebelumnya.")
        df_general = st.session_state['df_currency_raw_multi']

    if not df_general.empty:
        st.session_state['df_currency_raw_multi'] = df_general

        # Filter kolom numerik
        available_cols = [col for col in df_general.columns if pd.api.types.is_numeric_dtype(df_general[col])]

        if available_cols:
            current_idx = 0
            if st.session_state['selected_currency'] in available_cols:
                current_idx = available_cols.index(st.session_state['selected_currency'])

            st.session_state['selected_currency'] = st.selectbox("Pilih mata uang untuk analisis: 🎯", available_cols, index=current_idx, key="currency_selector")

            if st.session_state['selected_currency']:
                st.session_state['df_currency_raw'] = df_general[[st.session_state['selected_currency']]].rename(
                    columns={st.session_state['selected_currency']: 'Value'}
                )
                st.session_state['df_currency_raw'] = st.session_state['df_currency_raw'].sort_index()  # ⬅️ URUTKAN TANGGAL LAGI

                if st.session_state['variable_name'] == "Nama Variabel":
                    st.session_state['variable_name'] = st.session_state['selected_currency']

                with col2:
                    st.text_input("Jumlah Data:", value=str(len(st.session_state['df_currency_raw'])), disabled=True)
                    if isinstance(st.session_state['df_currency_raw'].index, pd.DatetimeIndex):
                        start_date = st.session_state['df_currency_raw'].index.min().strftime('%Y-%m-%d')
                        end_date = st.session_state['df_currency_raw'].index.max().strftime('%Y-%m-%d')
                        st.text_input("Tanggal Awal:", value=start_date, disabled=True)
                        st.text_input("Tanggal Akhir:", value=end_date, disabled=True)
                    else:
                        st.text_input("Tanggal Awal:", value="N/A (bukan indeks waktu)", disabled=True)
                        st.text_input("Tanggal Akhir:", value="N/A (bukan indeks waktu)", disabled=True)

        else:
            st.warning("🚫 Tidak ada kolom numerik terdeteksi. Pastikan data nilai tukar bertipe angka.")
            st.session_state['df_currency_raw'] = pd.DataFrame()

        with col2:
            st.text_input("Jumlah Data:", value="0", disabled=True)
            st.text_input("Tanggal Awal:", value="N/A", disabled=True)
            st.text_input("Tanggal Akhir:", value="N/A", disabled=True)

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        st.subheader(f"📊 Data Terpilih: {st.session_state['selected_currency']}")
        st.dataframe(st.session_state['df_currency_raw'])

        st.subheader(f"📈 Grafik Nilai Tukar: {st.session_state['selected_currency']}")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(
            x=st.session_state['df_currency_raw'].index,
            y=st.session_state['df_currency_raw']['Value'],
            mode='lines+markers',
            name='Nilai Tukar',
            line=dict(color='#1f77b4', width=2)
        ))
        fig_raw.update_layout(
            title=f'Grafik Nilai Tukar {st.session_state["selected_currency"]}',
            xaxis_title='Tanggal',
            yaxis_title='Nilai Tukar',
            template='plotly_white',
            xaxis_rangeslider_visible=True
        )
        st.plotly_chart(fig_raw, use_container_width=True)

elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">Data Preprocessing, Splitting & Stasioneritas ⚙️✂️🧪</div>', unsafe_allow_html=True)
    st.write("Lakukan pembersihan, transformasi, pembagian data, dan analisis stasioneritas nilai tukar.")

    #Cek data mentah
    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        df_raw = st.session_state['df_currency_raw'].copy()
        st.write(f"Data nilai tukar mentah untuk {st.session_state.get('selected_currency', '')}: 📊")
        st.dataframe(df_raw.head())

        st.subheader("Pilih Kolom Data dan Transformasi!")

        #Dropdown untuk memilih kolom numerik
        numeric_cols = [col for col in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[col])]
        
        if numeric_cols:
            selected_column = st.selectbox(
                "Pilih kolom data nilai tukar yang akan diproses:",
                options=numeric_cols,
                key="selected_column"
            )
            series_data = df_raw[selected_column]
            
            # Transformasi ke Log-Return
            st.markdown("##### Transformasi: Log-Return 📉")
            apply_log_return = st.checkbox("Hitung log-return", value=True)

            log_return_series = None
            if apply_log_return:
                try:
                    if series_data.max() > 100000:
                        series_data = series_data / 1000
                        st.info("Skala data dibagi 1000 agar log-return lebih presisi.")

                    # Hitung log-return
                    log_return_series = np.log(series_data).diff().dropna()
                    st.session_state['log_return_series'] = log_return_series

                    if 'log_return_original' not in st.session_state:
                        st.session_state['log_return_original'] = log_return_series.copy()

                    st.success("Log-return berhasil dihitung dan disimpan di sesi. ✅")
                    st.write("📉 Grafik Log-Return:")
                    st.line_chart(log_return_series)

                    # Gabungkan dengan data aslinya (jika ingin preview berdampingan)
                    log_return_df = pd.concat([series_data, log_return_series], axis=1)
                    log_return_df.columns = ['Value', 'Log-Return']
                    st.write("🧾 Tabel Data dan Log-Return (5 data pertama):")
                    st.dataframe(log_return_df.dropna().head())

                    # Simpan ke session
                    st.session_state['log_return_df'] = log_return_df

                except Exception as e:
                    st.error(f"Gagal menghitung log-return: {e}")
                    st.stop()
            else:
                st.warning("Silahkan centang opsi log-return untuk melanjutkan.")
        
            # === SPLITTING ===
            st.markdown('<div class="main-header">Data Splitting ✂️📊</div>', unsafe_allow_html=True)
            st.info("📌 30 observasi terakhir digunakan sebagai data uji.")
            if st.button("Lakukan Pembagian Data ▶️", key="split_data_button"):
                log_return_series = st.session_state['log_return_original']
                log_return_series = log_return_series.sort_index()

                train = log_return_series.iloc[:-30]
                test = log_return_series.iloc[-30:]

                st.session_state['log_return_train'] = train
                st.session_state['log_return_test'] = test
                st.session_state['train_data_returns'] = train
                st.session_state['test_data_returns'] = test

                st.success("✅ Data berhasil dibagi menjadi Train dan Test.")
                st.write(f"Periode Train: {train.index.min().strftime('%d %b %Y')} – {train.index.max().strftime('%d %b %Y')}")
                st.write(f"Periode Test: {test.index.min().strftime('%d %b %Y')} – {test.index.max().strftime('%d %b %Y')}")
                
                # Visualisasi
                fig_split = go.Figure()
                fig_split.add_trace(go.Scatter(x=train.index, y=train.values, mode='lines', name='Train', line=dict(color='#3f72af')))
                fig_split.add_trace(go.Scatter(x=test.index, y=test.values, mode='lines', name='Test', line=dict(color='#ff7f0e')))
                fig_split.update_layout(title='Train/Test Split Log-Return', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_split)

            # === ADF TEST ===
            st.markdown('<div class="main-header">Stasioneritas Data 📊🧪</div>', unsafe_allow_html=True)
            adf_results_path = "models/adf_test_results_clean.pkl"

            # Load hasil ADF
            try:
                with open(adf_results_path, "rb") as f:
                    adf_results = pickle.load(f)
                st.success("✅ Hasil ADF Test berhasil dimuat.")
            except Exception as e:
                st.error(f"Gagal menampilkan hasil ADF Test: {e}")
                st.stop()

            # Tampilkan hasil ADF
            selected_currency = st.session_state.get("selected_currency", "")
            if selected_currency in adf_results:
                adf_stat = adf_results[selected_currency]["adf_stat"]
                p_value = adf_results[selected_currency]["p_value"]
                
                st.write(f"**Hasil ADF untuk {selected_currency}:**")
                st.write(f"• Statistik ADF: `{adf_stat:.6f}`")
                st.write(f"• P-value: `{p_value:.6f}`")
                if p_value < 0.05:
                    st.success("➡️ Data stasioner (tolak H₀)")
                else:
                    st.warning("➡️ Data tidak stasioner (gagal tolak H₀)")
            else:
                st.warning(f"Tidak ada hasil ADF untuk {selected_currency}.")
            
            # === ACF & PACF ===
            st.markdown("### 📈 Plot ACF & PACF (Data Train)")
            log_return_train = st.session_state.get("log_return_train", None)

            if log_return_train is not None and not log_return_train.empty:
                lags = st.slider("Jumlah lags:", 5, 50, 20, key="acf_pacf_lags_slider")
                if st.button("📊 Tampilkan ACF & PACF", key="show_acf_pacf_button"):
                    try:
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                        plot_acf(log_return_train, lags=lags, alpha=0.05, ax=axes[0])
                        axes[0].set_title(f"ACF {selected_currency} Log-Return (Train)")

                        plot_pacf(log_return_train, lags=lags, alpha=0.05, ax=axes[1])
                        axes[1].set_title(f"PACF {selected_currency} Log-Return (Train)")

                        fig.suptitle(f"ACF & PACF - {selected_currency} Log-Return", fontsize=14)
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.success("✅ Plot ACF & PACF berhasil ditampilkan.")
                    except Exception as e:
                        st.error(f"Gagal membuat plot ACF/PACF: {e}")
            else:
                st.warning("📛 Data train log-return belum tersedia. Silakan bagi data terlebih dahulu.")

        else:
            st.warning("Tidak ditemukan kolom numerik dalam file.")
    else:
        st.warning("Silakan unggah data terlebih dahulu. 📂")
   

elif st.session_state['current_page'] == 'ARIMA Model':
    st.markdown('<div class="main-header">MODEL ARIMA 📈</div>', unsafe_allow_html=True)
    st.write(f"Bangun dan evaluasi model ARIMA pada data log-return mata uang **{st.session_state.get('selected_currency', '')}**.")

    # Ambil data train dari log-return
    if 'log_return_train' in st.session_state and not st.session_state['log_return_train'].empty:
        train_data_returns = st.session_state['log_return_train']
        st.write("📊 Data pelatihan log-return (Train):")
        st.dataframe(train_data_returns.head())
    else:
        st.warning("📛 Data log-return pelatihan belum tersedia. Silakan lakukan preprocessing dan splitting terlebih dahulu.")
        st.stop()

    # 1. Input Ordo
    st.subheader("1. Tentukan Ordo ARIMA (p, d, q) 🔢")
    p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
    d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=0, value=0, key="arima_d")  # d=0 karena log-return
    q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

    if st.button("▶️ Latih Model ARIMA"):
        try:
            with st.spinner("Melatih model ARIMA..."):
                model_arima = ARIMA(train_data_returns, order=(p, d, q))
                model_arima_fit = model_arima.fit()

                st.session_state['model_arima_fit'] = model_arima_fit
                st.session_state['arima_residuals'] = model_arima_fit.resid.dropna()

                st.success("✅ Model ARIMA berhasil dilatih!")

                # 2. Ringkasan Model
                st.subheader("2. Ringkasan Model ARIMA")
                st.text(model_arima_fit.summary().as_text())

                # 3. Uji Signifikansi Koefisien
                st.subheader("3. Uji Signifikansi Koefisien")
                df_results = pd.read_html(model_arima_fit.summary().tables[1].as_html(), header=0, index_col=0)[0]
                st.dataframe(df_results[['P>|z|']].style.applymap(
                    lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))

                # 4. Uji Asumsi Residual
                st.subheader("4. Uji Asumsi Residual ARIMA")
                resid = model_arima_fit.resid.dropna()

                # Plot Residual
                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(x=resid.index, y=resid, mode='lines', name='Residual ARIMA'))
                fig_res.update_layout(title_text='Residual ARIMA', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_res)

                # Kolmogorov-Smirnov Test
                standardized_resid = (resid - np.mean(resid)) / np.std(resid)
                ks_stat, ks_pvalue = kstest(standardized_resid, 'norm')
                st.write(f"**Kolmogorov-Smirnov Test:**")
                st.write(f"• Statistik: `{ks_stat:.4f}`")
                st.write(f"• P-value: `{ks_pvalue:.4f}`")
                if ks_pvalue > 0.05:
                    st.success("Residual terdistribusi normal (Gagal tolak H₀)")
                else:
                    st.warning("Residual tidak normal (Tolak H₀)")

                # Ljung-Box Test
                lb_test = acorr_ljungbox(resid, lags=[10], return_df=True)
                st.write("**Ljung-Box Test:**")
                st.dataframe(lb_test)
                if lb_test['lb_pvalue'].iloc[0] > 0.05:
                    st.success("Tidak ada autokorelasi signifikan (Gagal tolak H₀)")
                else:
                    st.warning("Terdapat autokorelasi signifikan (Tolak H₀)")

                # ARCH Test
                lb_arch = acorr_ljungbox(resid**2, lags=[10], return_df=True)
                st.write("**ARCH Test (Ljung-Box pada residual kuadrat):**")
                st.dataframe(lb_arch)
                has_arch = lb_arch['lb_pvalue'].iloc[0] < 0.05
                if not has_arch:
                    st.success("Tidak ada efek ARCH signifikan (Gagal tolak H₀)")
                else:
                    st.warning("Ada efek ARCH signifikan (Tolak H₀)")

                # Simpan hasil uji ke file
                st.session_state['arima_residual_has_arch_effect'] = has_arch
                uji_asumsi = {
                    'ks_stat': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'ljung_box_stat': lb_test['lb_stat'].iloc[0],
                    'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[0],
                    'arch_stat': lb_arch['lb_stat'].iloc[0],
                    'arch_pvalue': lb_arch['lb_pvalue'].iloc[0],
                    'has_arch_effect': has_arch
                }

                mata_uang = st.session_state.get("selected_currency", "")
                file_name = f"models/uji_asumsi_arima_{mata_uang.lower()}.pkl"
                try:
                    with open(file_name, "wb") as f:
                        pickle.dump(uji_asumsi, f)
                    st.info(f"Hasil uji asumsi disimpan ke: `{file_name}`")
                except Exception as e:
                    st.warning(f"Gagal menyimpan hasil uji asumsi: {e}")

        except Exception as e:
            st.error(f"❌ Gagal melatih model ARIMA: {e}")

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
    

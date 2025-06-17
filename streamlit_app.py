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
    "MODEL ARIMA": "pemodelan_arima", # Diubah namanya
    "PREDIKSI ARIMA": "prediksi_arima", # Tambahan menu prediksi untuk ARIMA
    "MODEL NGARCH": "pemodelan_ngarch", # Diubah namanya
    "PREDIKSI NGARCH": "prediksi_ngarch", # Tambahan menu prediksi untuk NGARCH
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
        <li><b>MODEL ARIMA :</b> Langkah-langkah untuk membentuk model ARIMA pada data return (untuk prediksi nilai tukar), termasuk uji asumsi dan koefisien.</li>
        <li><b>PREDIKSI ARIMA :</b> Menampilkan hasil prediksi nilai tukar dari model ARIMA dan evaluasinya.</li>
        <li><b>MODEL NGARCH :</b> Langkah-langkah untuk membentuk model NGARCH pada residual ARIMA (untuk prediksi volatilitas), termasuk uji asumsi dan koefisien.</li>
        <li><b>PREDIKSI NGARCH :</b> Menampilkan hasil prediksi volatilitas dari model NGARCH dan visualisasinya.</li>
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
    st.write("Lakukan pembersihan dan transformasi data nilai tukar.✨")

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

        # Simpan hasil preprocessing ke session_state
        st.session_state['preprocessed_data'] = series_data

elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">STASIONERITAS DATA</div>', unsafe_allow_html=True)
    st.markdown("Menggunakan data hasil preprocessing.")
    
     if 'preprocessed_data' in st.session_state and not st.session_state['preprocessed_data'].empty:
         series_to_test = st.session_state['preprocessed_data'] 
         st.write(f"5 baris pertama data nilai tukar {st.session_state.get('selected_currency', '')} yang akan diuji:")
         st.dataframe(series_to_test.head())
        
        if st.button("Uji Stasioneritas", key="uji_adf"):
            series = df[kolom_pilihan].dropna()
            result = adfuller(series)
            stat, pval, _, _, crit_values, _ = result

            st.subheader("Uji ADF Awal")
            st.write(f"**ADF Statistic:** {stat:.4f}")
            st.write(f"**P-Value:** {pval:.4f}")
            st.write(f"**Critical Value (1%)**: {crit_values['1%']:.4f}")
            st.write(f"**Critical Value (5%)**: {crit_values['5%']:.4f}")
            st.write(f"**Critical Value (10%)**: {crit_values['10%']:.4f}")

            if pval < 0.05:
                st.success("🟩 Data sudah stasioner.")
                st.session_state['d_order'] = 0
            else:
                st.warning("🟥 Data belum stasioner. Disarankan melakukan differencing.")
                st.session_state['d_order'] = 1

            with st.expander("Keterangan 📌"):
                st.markdown("""
                - **P-Value < 0.05** → menolak H0 → **data stasioner**  
                - **d = 0** (tidak perlu differencing)
                """)
            
        # Plot ACF & PACF
        st.subheader("Plot ACF dan PACF:")
        lag = st.slider("Jumlah lag:", 5, 50, 20)
        if st.button("Tampilkan ACF & PACF"):
            fig1 = plot_acf(df[kolom_pilihan].dropna(), lags=lag)
            st.pyplot(fig1)
            fig2 = plot_pacf(df[kolom_pilihan].dropna(), lags=lag)
            st.pyplot(fig2)
    else:
        st.info("Silakan unggah dan preprocessing data terlebih dahulu.")

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
                        st.warning("Residual ARIMA kosong atau tidak valid untuk pengujian. ❌")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e} ❌ Pastikan data pelatihan tidak kosong dan ordo ARIMA sesuai.")
                st.info("Kesalahan umum: data terlalu pendek untuk ordo yang dipilih, atau ada nilai NaN/Inf.")
    else:
        st.info("Silakan bagi data terlebih dahulu di halaman 'Data Splitting' untuk melatih model ARIMA. ✂️")


elif st.session_state['current_page'] == 'prediksi_arima':
    st.markdown('<div class="main-header">PREDIKSI ARIMA (Nilai Tukar) 📈</div>', unsafe_allow_html=True)
    st.write(f"Gunakan model ARIMA yang sudah dilatih untuk memprediksi nilai tukar {st.session_state.get('selected_currency', '')} dan evaluasi performanya. 🚀")

    if 'model_arima_fit' in st.session_state and 'test_data_returns' in st.session_state and 'original_prices_for_reconstruction' in st.session_state:
        model_arima_fit = st.session_state['model_arima_fit']
        test_data_returns = st.session_state['test_data_returns']
        original_prices = st.session_state['original_prices_for_reconstruction']
        return_type = st.session_state['return_type']

        st.subheader("1. Prediksi Return dengan Model ARIMA 🔮")
        forecast_steps = len(test_data_returns)
        st.write(f"Melakukan prediksi untuk {forecast_steps} langkah ke depan (sesuai ukuran data pengujian).")

        try:
            arima_forecast_returns = model_arima_fit.predict(start=test_data_returns.index[0], end=test_data_returns.index[-1])
            st.session_state['arima_forecast_returns'] = arima_forecast_returns
            st.success("Prediksi return dengan ARIMA berhasil! 🎉")
            st.write("5 nilai prediksi return pertama:")
            st.dataframe(arima_forecast_returns.head())

            st.subheader("2. Rekonstruksi Prediksi Nilai Tukar dari Return 🔄")
            st.info("Prediksi return perlu diubah kembali menjadi prediksi nilai tukar mata uang agar mudah diinterpretasikan.")

            last_train_price = original_prices.loc[model_arima_fit.fittedvalues.index[-1]]

            # Menyelaraskan indeks untuk rekonstruksi
            # Pastikan original_prices mencakup seluruh periode hingga akhir data pelatihan
            # dan prediksi dimulai dari observasi pertama data test
            
            # Buat series kosong untuk menyimpan harga yang direkonstruksi
            reconstructed_prices = pd.Series(index=arima_forecast_returns.index, dtype=float)
            
            # Nilai awal untuk rekonstruksi adalah harga terakhir dari data pelatihan
            previous_price = last_train_price

            for i, (date, forecast_return) in enumerate(arima_forecast_returns.items()):
                if return_type == "Log Return":
                    current_predicted_price = previous_price * np.exp(forecast_return)
                else: # Simple Return
                    current_predicted_price = previous_price * (1 + forecast_return)
                reconstructed_prices.loc[date] = current_predicted_price
                previous_price = current_predicted_price # Update previous_price for next step

            st.session_state['arima_forecast_prices'] = reconstructed_prices
            st.success("Prediksi nilai tukar berhasil direkonstruksi! 🎉")
            st.write("5 nilai prediksi nilai tukar pertama:")
            st.dataframe(reconstructed_prices.head())

            st.subheader("3. Visualisasi Prediksi Nilai Tukar ARIMA vs. Aktual 📊")
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

            st.subheader("4. Evaluasi Model ARIMA 🧪")
            st.info("Metrik evaluasi seperti RMSE, MAE, dan MAPE digunakan untuk mengukur akurasi prediksi.")

            # Menyelaraskan indeks untuk evaluasi
            actual_prices_test = original_prices.loc[test_data_returns.index[0]:test_data_returns.index[-1]]
            
            # Pastikan indeks prediksi dan aktual sama persis
            common_index = actual_prices_test.index.intersection(reconstructed_prices.index)
            actual_prices_test_aligned = actual_prices_test.loc[common_index]
            predicted_prices_aligned = reconstructed_prices.loc[common_index]

            if not actual_prices_test_aligned.empty:
                # RMSE
                rmse_arima = np.sqrt(np.mean((predicted_prices_aligned - actual_prices_test_aligned)**2))
                st.write(f"**RMSE (Root Mean Squared Error):** {rmse_arima:.4f}")

                # MAE
                mae_arima = np.mean(np.abs(predicted_prices_aligned - actual_prices_test_aligned))
                st.write(f"**MAE (Mean Absolute Error):** {mae_arima:.4f}")

                # MAPE
                # Menghindari pembagian oleh nol
                mape_arima = np.mean(np.abs((actual_prices_test_aligned - predicted_prices_aligned) / actual_prices_test_aligned.replace(0, np.nan))) * 100
                st.write(f"**MAPE (Mean Absolute Percentage Error):** {mape_arima:.2f}%")
            else:
                st.warning("Tidak ada data aktual yang cocok untuk evaluasi. Pastikan rentang indeks data uji sesuai dengan prediksi. 🤷")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memprediksi atau merekonstruksi nilai tukar dengan ARIMA: {e} ❌")
            st.info("Pastikan model ARIMA sudah dilatih dengan benar dan data test tersedia.")
    else:
        st.info("Silakan latih model ARIMA di halaman 'Model ARIMA' dan pastikan data splitting sudah dilakukan. 📈✂️")


elif st.session_state['current_page'] == 'pemodelan_ngarch':
    st.markdown('<div class="main-header">MODEL NGARCH (Volatility Equation) 🌪️</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual kuadrat dari model ARIMA untuk memodelkan volatilitas {st.session_state.get('selected_currency', '')}. Ini penting untuk memahami risiko. 💥")

    if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
        arima_residuals = st.session_state['arima_residuals']
        st.write(f"Data residual ARIMA ({st.session_state.get('selected_currency', '')}) yang akan digunakan untuk model NGARCH:")
        st.dataframe(arima_residuals.head())

        if 'arima_residual_has_arch_effect' in st.session_state and st.session_state['arima_residual_has_arch_effect'] == False:
            st.warning("Berdasarkan Uji Heteroskedastisitas (Ljung-Box pada Residual Kuadrat) di 'MODEL ARIMA', residual ARIMA **tidak menunjukkan efek ARCH/GARCH signifikan**. 😟")
            st.info("Meskipun demikian, Anda masih bisa melanjutkan melatih model NGARCH, tetapi hasilnya mungkin tidak seefektif jika ada efek ARCH/GARCH yang jelas. Ini mungkin berarti volatilitas konstan atau sudah ditangkap oleh model ARIMA.")
        else:
            st.info("Berdasarkan Uji Heteroskedastisitas (Ljung-Box pada Residual Kuadrat) di 'MODEL ARIMA', residual ARIMA **menunjukkan efek ARCH/GARCH signifikan**. Model NGARCH sangat cocok di sini! 👍")

        st.subheader("1. Tentukan Ordo NGARCH (p, q) 🔢")
        st.info("Untuk NGARCH(p, q), 'p' adalah ordo ARCH (jumlah lag dari residual kuadrat) dan 'q' adalah ordo GARCH (jumlah lag dari varians bersyarat). Umumnya GARCH(1,1) adalah titik awal yang baik.")
        ngarch_p = st.number_input("Ordo ARCH (p):", min_value=1, max_value=5, value=1, key="ngarch_p")
        ngarch_o = st.number_input("Ordo Asymmetric (o):", min_value=0, max_value=1, value=1, help="Ordo asimetris untuk efek leverage. Set ke 0 untuk GARCH biasa. Set ke 1 untuk NGARCH/GJR-GARCH.", key="ngarch_o")
        ngarch_q = st.number_input("Ordo GARCH (q):", min_value=1, max_value=5, value=1, key="ngarch_q")

        if st.button("2. Latih Model NGARCH ▶️", key="train_ngarch_button"):
            try:
                # Ambil residual sebagai seri
                returns_for_ngarch = arima_residuals.dropna()

                if returns_for_ngarch.empty:
                    st.error("Residual ARIMA kosong atau hanya berisi NaN setelah pembersihan. Tidak dapat melatih NGARCH. ❌")
                    st.stop()

                with st.spinner("Melatih model NGARCH... ⏳"):
                    # mean='zero' karena kita memodelkan residual (yang diharapkan memiliki mean nol)
                    # p dan q untuk GARCH, o untuk asimetri (NGARCH)
                    # dist='t' atau 'skewt' sering digunakan untuk data keuangan karena sifat fat tails
                    ngarch_model = arch_model(
                        returns_for_ngarch,
                        mean='zero',
                        vol='Garch',
                        p=ngarch_p,
                        o=ngarch_o,
                        q=ngarch_q,
                        dist='t' # Menggunakan distribusi Student's t untuk menangani fat tails
                    )
                    ngarch_fit = ngarch_model.fit(disp='off') # disp='off' untuk menonaktifkan output verbose

                    st.session_state['model_ngarch_fit'] = ngarch_fit
                    st.success("Model NGARCH berhasil dilatih! 🎉")
                    st.subheader("3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                    st.text(ngarch_fit.summary().as_text())

                    st.subheader("4. Uji Signifikansi Koefisien NGARCH (P-value) ✅❌")
                    st.info("Sama seperti ARIMA, koefisien dianggap signifikan jika P-value < 0.05. Fokus pada koefisien $\\omega$, $\\alpha_i$, $\\gamma_i$, dan $\\beta_i$.")
                    
                    # Ekstrak tabel parameter
                    # arch_model summary uses statsmodels-like structure for summary tables
                    # The parameters table is usually the second table
                    results_html = ngarch_fit.summary().as_html()
                    df_ngarch_results = pd.read_html(results_html, header=0, index_col=0)[0]
                    
                    # Filter for relevant columns and display
                    if 'P>|z|' in df_ngarch_results.columns:
                        st.dataframe(df_ngarch_results[['P>|z|']].style.applymap(lambda x: 'background-color: #d4edda' if x < 0.05 else 'background-color: #f8d7da'))
                        st.caption("Hijau: Signifikan (P < 0.05), Merah: Tidak Signifikan (P >= 0.05)")
                    else:
                        st.warning("Kolom 'P>|z|' tidak ditemukan dalam ringkasan model NGARCH. Tidak dapat menampilkan signifikansi koefisien. 🤷")

                    st.subheader("5. Uji Asumsi Residual Standar NGARCH 📊")
                    # Residual standar adalah residual dibagi dengan estimasi standar deviasi bersyarat
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
    else:
        st.info("Silakan latih model ARIMA dan dapatkan residualnya di halaman 'Model ARIMA' terlebih dahulu. 📈")


elif st.session_state['current_page'] == 'prediksi_ngarch':
    st.markdown('<div class="main-header">PREDIKSI NGARCH (Volatilitas) 🌪️</div>', unsafe_allow_html=True)
    st.write(f"Gunakan model NGARCH yang sudah dilatih untuk memprediksi volatilitas bersyarat {st.session_state.get('selected_currency', '')}. 📊")

    if 'model_ngarch_fit' in st.session_state and 'test_data_returns' in st.session_state:
        ngarch_fit = st.session_state['model_ngarch_fit']
        test_data_returns = st.session_state['test_data_returns']

        st.subheader("1. Prediksi Volatilitas dengan Model NGARCH 🚀")
        forecast_steps = len(test_data_returns)
        st.write(f"Melakukan prediksi volatilitas untuk {forecast_steps} langkah ke depan.")

        try:
            # Prediksi conditional variance (varians bersyarat)
            ngarch_forecast_results = ngarch_fit.forecast(horizon=forecast_steps, start=test_data_returns.index[0])
            
            # Ambil prediksi varians dari horizon 1 (prediksi 1 langkah ke depan)
            # .variance attribute provides a DataFrame of variances, we need the first column (horizon 1)
            # and slice to match the test data length.
            # Make sure the index of the forecast aligns with the test_data_returns index
            
            # The forecast method's `variance` attribute returns a DataFrame where columns are horizons.
            # For 1-step ahead prediction for the test set, we often look at the first horizon (h.1).
            # The index will usually be the start of the forecast.
            
            # Let's verify the output structure of ngarch_fit.forecast
            # It returns an ARCHModelForecast object which has .variance, .mean, .residual_variance
            # ngarch_forecast_results.variance.values will give a numpy array of forecasts
            # We need to correctly map these to the dates in test_data_returns.
            
            # Conditional volatility (standard deviation) is sqrt of conditional variance
            predicted_volatility = np.sqrt(ngarch_forecast_results.variance.dropna().iloc[0, :forecast_steps])
            
            # Create a Series with the correct index
            # The dates for the forecast should correspond to the dates in test_data_returns
            predicted_volatility.index = test_data_returns.index[:forecast_steps]

            st.session_state['ngarch_forecast_volatility'] = predicted_volatility
            st.success("Prediksi volatilitas dengan NGARCH berhasil! 🎉")
            st.write("5 nilai prediksi volatilitas pertama:")
            st.dataframe(predicted_volatility.head())

            st.subheader("2. Visualisasi Prediksi Volatilitas Bersyarat NGARCH 📊")
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
                x=predicted_volatility.index,
                y=predicted_volatility.values,
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
            st.subheader("3. Perbandingan dengan Volatilitas Aktual (Squared Returns) 📉")
            st.info("Volatilitas aktual tidak dapat diamati secara langsung, tetapi kuadrat dari return adalah proksi yang umum digunakan untuk memvisualisasikan volatilitas historis.")
            
            fig_actual_vs_pred_vol = go.Figure()
            
            # Squared returns for the whole series (train + test)
            actual_squared_returns = (st.session_state['processed_returns']**2)
            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=actual_squared_returns.index,
                y=actual_squared_returns.values,
                mode='lines',
                name='Kuadrat Return Aktual',
                line=dict(color='#8c564b', opacity=0.7)
            ))

            # Conditional volatility (in-sample)
            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=conditional_vol_train.index,
                y=conditional_vol_train.values**2, # Square it for comparison with squared returns
                mode='lines',
                name='Varians Bersyarat (In-Sample)',
                line=dict(color='#2ca02c', width=2)
            ))

            # Predicted volatility (out-of-sample) squared
            fig_actual_vs_pred_vol.add_trace(go.Scatter(
                x=predicted_volatility.index,
                y=predicted_volatility.values**2, # Square it for comparison with squared returns
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
        st.markdown(f"Model ARIMA({arima_fit.k_ar},0,{arima_fit.k_ma}) telah dilatih pada data return {st.session_state.get('selected_currency', '')}.")
        st.markdown("Berikut adalah beberapa poin penting dari ringkasan model:")
        
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
        st.markdown(f"Model NGARCH({ngarch_fit.p},{ngarch_fit.o},{ngarch_fit.q}) dengan distribusi residual Student's t telah dilatih pada residual ARIMA.")
        
        st.markdown("Berikut adalah beberapa poin penting dari ringkasan model:")
        
        # Check for significant coefficients
        results_html_ngarch = ngarch_fit.summary().as_html()
        df_ngarch_results = pd.read_html(results_html_ngarch, header=0, index_col=0)[0]
        
        significant_params_ngarch = df_ngarch_results[df_ngarch_results['P>|z|'] < 0.05]
        insignificant_params_ngarch = df_ngarch_results[df_ngarch_results['P>|z|'] >= 0.05]

        if not significant_params_ngarch.empty:
            st.success("✅ **Koefisien Signifikan:**")
            for index, row in significant_params_ngarch.iterrows():
                st.write(f"- Parameter `{index}` (P-value: {row['P>|z|']:.4f}) signifikan secara statistik.")
                if index.startswith('alpha'):
                    st.write("  - Ini menunjukkan adanya efek ARCH (volatilitas saat ini dipengaruhi oleh ukuran kejutan masa lalu).")
                elif index.startswith('gamma'):
                    st.write("  - Ini menunjukkan adanya efek leverage (berita buruk/kejutan negatif memiliki dampak yang lebih besar pada volatilitas dibandingkan berita baik/kejutan positif).")
                elif index.startswith('beta'):
                    st.write("  - Ini menunjukkan adanya efek GARCH (volatilitas saat ini dipengaruhi oleh volatilitas masa lalu), yang berarti volatilitas berkelompok.")
                elif index == 'omega':
                    st.write("  - Ini adalah konstanta varians bersyarat.")
                elif index == 'nu': # For Student's t-distribution degrees of freedom
                    st.write("  - Ini adalah derajat kebebasan distribusi Student's t, menunjukkan *fatness* ekor distribusi residual.")

        else:
            st.warning("⚠️ **Tidak Ada Koefisien Signifikan:** Tidak ada koefisien ARCH, GARCH, atau Leverage yang signifikan pada tingkat 5%. Ini mungkin menunjukkan bahwa model NGARCH tidak sepenuhnya diperlukan atau efek volatilitas sudah ditangkap oleh ARIMA.")
        
        # Check Ljung-Box for NGARCH squared standard residuals
        if 'ngarch_std_residuals' in st.session_state and not st.session_state['ngarch_std_residuals'].empty:
            std_residuals = st.session_state['ngarch_std_residuals']
            lb_arch_test_ngarch = sm.stats.acorr_ljungbox(std_residuals.dropna()**2, lags=[10], return_df=True)
            if lb_arch_test_ngarch['lb_pvalue'].iloc[0] > 0.05:
                st.success("✅ **Residual Standar Kuadrat NGARCH:** Tidak menunjukkan autokorelasi signifikan, yang berarti model NGARCH telah berhasil menangkap volatilitas berkelompok (clustering).")
            else:
                st.warning("⚠️ **Residual Standar Kuadrat NGARCH:** Masih menunjukkan autokorelasi signifikan. Ini mengindikasikan bahwa model NGARCH mungkin perlu disempurnakan (misalnya, dengan mengubah ordo atau mencoba model GARCH lain seperti EGARCH) untuk menangkap sepenuhnya dinamika volatilitas.")
        else:
            st.info("Informasi residual standar NGARCH tidak tersedia.")

    else:
        st.info("Model NGARCH belum dilatih. Silakan kunjungi halaman 'Model NGARCH'.")

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

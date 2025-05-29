import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go # Menambahkan Plotly untuk visualisasi interaktif
from plotly.subplots import make_subplots # Untuk subplots di Plotly
pip install --upgrade statsmodels
pip install --upgrade pandas numpy scikit-learn matplotlib plotly arch statsmodels

# Impor model yang relevan
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model # Untuk GARCH/NGARCH

# --- Konfigurasi Halaman (Hanya dipanggil sekali di awal) ---
st.set_page_config(
    page_title='Prediksi ARIMA-NGARCH Volatilitas Mata Uang',
    page_icon='📈',
    layout="wide"
)

# --- Fungsi Pembaca Data (dengan caching) ---
@st.cache_data(ttl=86400)
def load_data(uploaded_file=None, default_filename='data/default_currency.csv'):
    """
    Membaca data dari objek file yang diunggah atau dari file default lokal.
    Untuk data mata uang, asumsikan kolom pertama adalah nilai tukar.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Asumsi kolom pertama adalah data time series
            df.columns = ['Value'] # Beri nama kolom untuk memudahkan
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file yang diunggah: {e}")
            return pd.DataFrame()
    else:
        path = Path(__file__).parent / default_filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = ['Value'] # Beri nama kolom untuk memudahkan
                return df
            except Exception as e:
                st.warning(f"Tidak dapat membaca file default '{default_filename}': {e}")
                return pd.DataFrame()
        else:
            st.warning(f"File default '{default_filename}' tidak ditemukan di {path}. Silakan unggah file.")
            return pd.DataFrame()

# --- Custom CSS untuk Tampilan ---
st.markdown("""
    <style>
        /* Mengubah warna latar belakang sidebar */
        .css-1d3f8aq.e1fqkh3o1 {
            background-color: #f0f2f6;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Mengatur padding untuk konten utama */
        .css-1v0mbdj.e1fqkh3o0 {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        /* Styling untuk tombol di sidebar */
        .stButton>button {
            width: 100%;
            border-radius: 0.5rem;
            border: 1px solid #ddd;
            background-color: #f0f2f6;
            color: #333;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            text-align: left;
            margin-bottom: 0.2rem;
        }
        .stButton>button:hover {
            background-color: #e0e2e6;
        }
        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }
        /* Styling untuk tombol aktif (klik) */
        .stButton>button:active {
            background-color: #c0c2c6;
        }
        .stButton button[data-testid^="stSidebarNavButton"]:focus:not(:active) {
            background-color: #d0d2d6 !important;
            font-weight: bold;
        }
        .main-header {
            background-color: #c94f71;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.8em;
            margin-bottom: 2rem;
        }
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        .info-card {
            background-color: #ffffff;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            text-align: center;
        }
        .info-card .plus-icon {
            font-size: 4em;
            color: #c94f71;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Menu ---
st.sidebar.markdown("#### MENU NAVIGASI")

menu_items = {
    "HOME": "home",
    "INPUT DATA": "input_data",
    "DATA PREPROCESSING": "data_preprocessing",
    "STASIONERITAS DATA": "stasioneritas_data",
    "DATA SPLITTING": "data_splitting",
    "PEMODELAN ARIMA": "pemodelan_arima", # Tetap ARIMA untuk nilai tukar
    "PEMODELAN NGARCH": "pemodelan_ngarch", # Mengganti ANFIS ABC menjadi NGARCH
    "PEMODELAN ARIMA-NGARCH": "pemodelan_arima_ngarch", # Mengganti ARIMA-ANFIS ABC
    "PREDIKSI": "prediksi",
}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# --- Area Konten Utama Berdasarkan Halaman yang Dipilih ---

if st.session_state['current_page'] == 'home':
    st.markdown('<div class="main-header">Prediksi Data Time Series Univariat <br> Menggunakan Model ARIMA-NGARCH</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <div class="plus-icon">+</div>
            <p>Sistem ini dirancang untuk melakukan prediksi nilai tukar mata uang menggunakan model ARIMA dan mengukur volatilitasnya dengan model NGARCH.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">Panduan Penggunaan Sistem</h3>', unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><b>HOME:</b> Halaman utama yang menjelaskan tujuan dan metode prediksi sistem.</li>
        <li><b>INPUT DATA:</b> Unggah data time series nilai tukar mata uang.</li>
        <li><b>DATA PREPROCESSING:</b> Lakukan pembersihan dan transformasi data (misalnya, menghitung return).</li>
        <li><b>STASIONERITAS DATA:</b> Uji stasioneritas data return sebelum model ARIMA dibentuk.</li>
        <li><b>DATA SPLITTING:</b> Pisahkan data menjadi latih dan uji.</li>
        <li><b>PEMODELAN ARIMA:</b> Langkah-langkah untuk membentuk model ARIMA pada data return.</li>
        <li><b>PEMODELAN NGARCH:</b> Langkah-langkah untuk membentuk model NGARCH pada residual ARIMA untuk mengukur volatilitas.</li>
        <li><b>PEMODELAN ARIMA-NGARCH:</b> Integrasi model ARIMA dan NGARCH.</li>
        <li><b>PREDIKSI:</b> Menampilkan hasil prediksi nilai tukar dan volatilitas.</li>
    </ul>
    """, unsafe_allow_html=True)

elif st.session_state['current_page'] == 'input_data':
    st.markdown('<div class="main-header">Input Data</div>', unsafe_allow_html=True)
    st.write("Di sinilah Anda dapat mengunggah data time series nilai tukar mata uang. Pastikan file CSV hanya memiliki satu kolom data numerik (harga atau nilai tukar).")

    uploaded_file_input_data_page = st.file_uploader("Pilih file CSV data nilai tukar", type="csv", key="input_data_uploader")

    df_general = pd.DataFrame()
    if uploaded_file_input_data_page is not None:
        try:
            df_general = load_data(uploaded_file=uploaded_file_input_data_page)
            if not df_general.empty:
                st.success("File berhasil diunggah dan dibaca!")
                st.write("5 baris pertama dari data Anda:")
                st.dataframe(df_general.head())
                st.session_state['df_currency_raw'] = df_general # Simpan data mentah
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.warning("Pastikan file yang diunggah adalah file CSV yang valid dan hanya berisi satu kolom data numerik.")
    elif 'df_currency_raw' not in st.session_state or st.session_state['df_currency_raw'].empty:
        st.info("Tidak ada file yang diunggah. Mencoba memuat data default 'data/default_currency.csv'.")
        df_general = load_data(uploaded_file=None, default_filename='data/default_currency.csv')
        if not df_general.empty:
            st.success("Data default berhasil dimuat.")
            st.dataframe(df_general.head())
            st.session_state['df_currency_raw'] = df_general
        else:
            st.warning("Tidak ada data yang dimuat. Silakan unggah file atau pastikan 'data/default_currency.csv' ada dan tidak kosong.")
    else:
        st.write("Data nilai tukar yang sudah diunggah sebelumnya:")
        st.dataframe(st.session_state['df_currency_raw'].head())

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        st.subheader("Visualisasi Data Nilai Tukar Mentah")
        fig_raw = go.Figure()
        fig_raw.add_trace(go.Scatter(y=st.session_state['df_currency_raw']['Value'], mode='lines', name='Nilai Tukar'))
        fig_raw.update_layout(title_text='Grafik Nilai Tukar Mentah', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw)


elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">Data Preprocessing</div>', unsafe_allow_html=True)
    st.write("Lakukan pembersihan dan transformasi data nilai tukar. Untuk model ARIMA-NGARCH, kita perlu mengubah data harga menjadi return (perubahan logaritmik atau persentase).")

    if 'df_currency_raw' in st.session_state and not st.session_state['df_currency_raw'].empty:
        df_raw = st.session_state['df_currency_raw'].copy()
        st.write("Data nilai tukar mentah:")
        st.dataframe(df_raw.head())

        st.subheader("Pilih Kolom Data dan Transformasi")
        # Asumsi kolom 'Value' sudah ada dari load_data
        series_data = df_raw['Value']

        # --- Penanganan Missing Values ---
        st.markdown("##### Penanganan Missing Values")
        if series_data.isnull().any():
            st.warning(f"Terdapat nilai hilang ({series_data.isnull().sum()} nilai).")
            missing_strategy = st.selectbox("Pilih strategi penanganan missing values:",
                                            ["Drop NA", "Isi dengan Mean", "Isi dengan Median", "Isi dengan Nilai Sebelumnya (FFill)", "Isi dengan Nilai Berikutnya (BFill)"],
                                            key="missing_strategy")
            if missing_strategy == "Drop NA":
                series_data = series_data.dropna()
                st.info("Nilai hilang dihapus.")
            elif missing_strategy == "Isi dengan Mean":
                series_data = series_data.fillna(series_data.mean())
                st.info("Nilai hilang diisi dengan mean.")
            elif missing_strategy == "Isi dengan Median":
                series_data = series_data.fillna(series_data.median())
                st.info("Nilai hilang diisi dengan median.")
            elif missing_strategy == "Isi dengan Nilai Sebelumnya (FFill)":
                series_data = series_data.fillna(method='ffill')
                st.info("Nilai hilang diisi dengan nilai sebelumnya (forward fill).")
            elif missing_strategy == "Isi dengan Nilai Berikutnya (BFill)":
                series_data = series_data.fillna(method='bfill')
                st.info("Nilai hilang diisi dengan nilai berikutnya (backward fill).")
            else:
                st.info("Nilai hilang dibiarkan.")

        st.subheader("Transformasi Data: Harga ke Return")
        return_type = st.radio("Pilih tipe return:", ("Log Return", "Simple Return"), key="return_type_radio")

        if st.button("Hitung Return", key="calculate_return_button"):
            if len(series_data) > 1:
                if return_type == "Log Return":
                    processed_series = np.log(series_data / series_data.shift(1)).dropna()
                    st.info("Data telah diubah menjadi Log Return.")
                else: # Simple Return
                    processed_series = series_data.pct_change().dropna()
                    st.info("Data telah diubah menjadi Simple Return (Persentase Perubahan).")

                if not processed_series.empty:
                    st.session_state['processed_returns'] = processed_series
                    st.session_state['original_prices_for_reconstruction'] = series_data # Simpan harga asli untuk rekonstruksi
                    st.success("Data return berhasil dihitung.")
                    st.write("5 baris pertama data return:")
                    st.dataframe(processed_series.head())

                    st.subheader("Visualisasi Data Return")
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Scatter(y=processed_series, mode='lines', name='Data Return'))
                    fig_returns.update_layout(title_text='Grafik Data Return', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_returns)
                else:
                    st.warning("Data return kosong setelah transformasi. Pastikan data input Anda valid.")
            else:
                st.warning("Data terlalu pendek untuk menghitung return.")
        else:
            st.info("Klik 'Hitung Return' untuk melanjutkan.")

    else:
        st.info("Unggah data nilai tukar terlebih dahulu di bagian 'Input Data' untuk melakukan preprocessing.")

elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">Stasioneritas Data Return</div>', unsafe_allow_html=True)
    st.write("Untuk pemodelan ARIMA, data harus stasioner (mean, varians, dan autokorelasi konstan seiring waktu). Kita akan menguji stasioneritas pada data return.")

    if 'processed_returns' in st.session_state and not st.session_state['processed_returns'].empty:
        series_to_test = st.session_state['processed_returns']
        st.write("5 baris pertama data return yang akan diuji:")
        st.dataframe(series_to_test.head())

        st.subheader("Uji Augmented Dickey-Fuller (ADF)")
        if st.button("Jalankan Uji ADF", key="run_adf_test"):
            try:
                result_adf = adfuller(series_to_test)
                st.write(f"**Statistik ADF:** {result_adf[0]:.4f}")
                st.write(f"**P-value:** {result_adf[1]:.4f}")
                st.write(f"**Jumlah Lags Optimal:** {result_adf[2]}")
                st.write("**Nilai Kritis:**")
                for key, value in result_adf[4].items():
                    st.write(f"  {key}: {value:.4f}")

                if result_adf[1] <= 0.05:
                    st.success("Data return **stasioner** (tolak H0: ada akar unit). Ini baik untuk ARIMA.")
                    st.session_state['is_stationary'] = True # Menandai data stasioner
                else:
                    st.warning("Data return **tidak stasioner** (gagal tolak H0: ada akar unit).")
                    st.info("Meskipun data return seringkali stasioner, jika tidak, Anda mungkin perlu transformasi tambahan (misalnya, differencing pada return, yang jarang terjadi).")
                    st.session_state['is_stationary'] = False

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan Uji ADF: {e}")
                st.warning("Pastikan data numerik dan tidak memiliki nilai tak terbatas/NaN.")

        st.subheader("Uji Kwiatkowski-Phillips-Schmidt-Shin (KPSS)")
        if st.button("Jalankan Uji KPSS", key="run_kpss_test"):
            try:
                result_kpss = kpss(series_to_test, regression='c')
                st.write(f"**Statistik KPSS:** {result_kpss[0]:.4f}")
                st.write(f"**P-value:** {result_kpss[1]:.4f}")
                st.write(f"**Jumlah Lags Optimal:** {result_kpss[2]}")
                st.write("**Nilai Kritis:**")
                for key, value in result_kpss[3].items():
                    st.write(f"  {key}: {value:.4f}")

                if result_kpss[1] > 0.05:
                    st.success("Data return **stasioner** (gagal tolak H0: tidak ada akar unit).")
                    if 'is_stationary' in st.session_state:
                         st.session_state['is_stationary'] = st.session_state['is_stationary'] and True
                    else:
                        st.session_state['is_stationary'] = True
                else:
                    st.warning("Data return **tidak stasioner** (tolak H0: ada akar unit).")
                    st.session_state['is_stationary'] = False

            except Exception as e:
                st.error(f"Terjadi kesalahan saat menjalankan Uji KPSS: {e}")

        # Karena kita sudah bekerja dengan return (yang seharusnya stasioner),
        # differencing tambahan di sini mungkin tidak diperlukan untuk kebanyakan kasus.
        # Namun, jika tetap ingin ada opsi differencing, Anda bisa tambahkan.
        # Untuk kasus nilai tukar, biasanya return sudah stasioner.
    else:
        st.info("Silakan unggah dan proses data (hitung return) terlebih dahulu di halaman 'Input Data' dan 'Data Preprocessing'.")


elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">Data Splitting</div>', unsafe_allow_html=True)
    st.write("Pisahkan data return menjadi set pelatihan dan pengujian untuk melatih dan mengevaluasi model ARIMA. Pembagian akan dilakukan secara berurutan karena ini adalah data time series.")

    if 'processed_returns' in st.session_state and not st.session_state['processed_returns'].empty:
        data_to_split = st.session_state['processed_returns']
        st.write("Data return yang akan dibagi:")
        st.dataframe(data_to_split.head())

        st.subheader("Konfigurasi Pembagian Data")
        test_size_ratio = st.slider("Pilih rasio data pengujian (%):", 10, 50, 20, 5, key="test_size_slider")
        test_size_frac = test_size_ratio / 100.0
        st.write(f"Rasio pengujian: {test_size_ratio}% (Data pelatihan: {100 - test_size_ratio}%)")

        if st.button("Lakukan Pembagian Data", key="split_data_button"):
            train_size = int(len(data_to_split) * (1 - test_size_frac))
            train_data_returns = data_to_split.iloc[:train_size]
            test_data_returns = data_to_split.iloc[train_size:]

            st.session_state['train_data_returns'] = train_data_returns
            st.session_state['test_data_returns'] = test_data_returns

            st.success("Data return berhasil dibagi!")
            st.write(f"Ukuran data pelatihan: {len(train_data_returns)} sampel")
            st.write(f"Ukuran data pengujian: {len(test_data_returns)} sampel")

            st.subheader("Visualisasi Pembagian Data Return")
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(x=train_data_returns.index, y=train_data_returns.values, mode='lines', name='Data Pelatihan'))
            fig_split.add_trace(go.Scatter(x=test_data_returns.index, y=test_data_returns.values, mode='lines', name='Data Pengujian'))
            fig_split.update_layout(title_text='Pembagian Data Return Time Series', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_split)
    else:
        st.warning("Tidak ada data return yang tersedia untuk dibagi. Pastikan Anda telah melalui 'Input Data' dan 'Data Preprocessing'.")


elif st.session_state['current_page'] == 'pemodelan_arima':
    st.markdown('<div class="main-header">Pemodelan ARIMA untuk Nilai Tukar</div>', unsafe_allow_html=True)
    st.write("Latih model ARIMA pada data return untuk memodelkan mean (prediksi nilai tukar).")

    if 'train_data_returns' in st.session_state and not st.session_state['train_data_returns'].empty:
        train_data_returns = st.session_state['train_data_returns']
        st.write("Data pelatihan return untuk pemodelan ARIMA:")
        st.dataframe(train_data_returns.head())

        st.subheader("Pilih Ordo ARIMA (p, d, q)")
        # Untuk return, d seringkali 0 karena return cenderung stasioner
        p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
        d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=2, value=0, key="arima_d") # d=0 for returns
        q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

        if st.button("Latih Model ARIMA", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA..."):
                    # Pastikan data memiliki indeks yang berurutan
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.success("Model ARIMA berhasil dilatih!")
                    st.subheader("Ringkasan Model ARIMA")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("Residual Model ARIMA")
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(y=model_arima_fit.resid, mode='lines', name='Residual ARIMA'))
                    fig_res.update_layout(title_text='Residual Model ARIMA', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_res)

                    # Simpan residual untuk pemodelan NGARCH
                    st.session_state['arima_residuals'] = model_arima_fit.resid

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA: {e}")
                st.info("Pastikan data return Anda sesuai dan ordo ARIMA yang dipilih valid. Residual mungkin berisi NaN jika ada masalah konvergensi.")
    else:
        st.info("Silakan unggah, proses, dan bagi data terlebih dahulu di halaman 'Input Data', 'Data Preprocessing', dan 'Data Splitting'.")

elif st.session_state['current_page'] == 'pemodelan_ngarch':
    st.markdown('<div class="main-header">Pemodelan NGARCH untuk Volatilitas</div>', unsafe_allow_html=True)
    st.write("Latih model NGARCH pada residual dari model ARIMA untuk memodelkan volatilitas (varians bersyarat).")

    if 'arima_residuals' in st.session_state and not st.session_state['arima_residuals'].empty:
        arima_residuals = st.session_state['arima_residuals'].dropna() # Pastikan tidak ada NaN di residual
        st.write("Residual dari model ARIMA (data untuk model NGARCH):")
        st.dataframe(arima_residuals.head())

        if arima_residuals.empty:
            st.warning("Residual ARIMA kosong atau hanya berisi NaN. Pastikan model ARIMA berhasil dilatih dan menghasilkan residual yang valid.")
        else:
            st.subheader("Pilih Ordo NGARCH (p, o, q)")
            # NGARCH(p,o,q) di arch_model
            p_garch = st.number_input("Ordo ARCH (p):", min_value=1, max_value=3, value=1, key="ngarch_p")
            o_garch = st.number_input("Ordo Asymmetry (o):", min_value=0, max_value=2, value=1, key="ngarch_o") # o=0 untuk GARCH
            q_garch = st.number_input("Ordo GARCH (q):", min_value=1, max_value=3, value=1, key="ngarch_q")

            if st.button("Latih Model NGARCH", key="train_ngarch_button"):
                try:
                    with st.spinner("Melatih model NGARCH..."):
                        # vol='Garch' dengan parameter 'o' akan mengaktifkan model GARCH Asimetris (misal, NGARCH)
                        # Untuk GJR-GARCH atau EGARCH, gunakan 'Garch' atau 'EGARCH' dan sesuaikan parameternya
                        # NGARCH (Non-linear GARCH) secara eksplisit mungkin perlu implementasi manual atau pustaka spesifik
                        # arch_model dengan vol='Garch' dan parameter 'o' memberikan GJR-GARCH yang mirip NGARCH
                        # Kita akan menggunakan GJR-GARCH sebagai proxy untuk NGARCH jika NGARCH murni tidak ada.
                        # GJR-GARCH(p,o,q) biasanya p adalah ARCH, o adalah asymmetry, q adalah GARCH.
                        
                        model_ngarch = arch_model(arima_residuals, vol='Garch', p=p_garch, o=o_garch, q=q_garch, dist='t') # dist='t' sering lebih baik untuk financial data
                        res_ngarch = model_ngarch.fit(disp='off') # 'disp=off' untuk suppress output

                        st.session_state['model_ngarch_fit'] = res_ngarch
                        st.success("Model NGARCH (GJR-GARCH) berhasil dilatih!")
                        st.subheader("Ringkasan Model NGARCH")
                        st.text(res_ngarch.summary().as_text())

                        st.subheader("Volatilitas Kondisional (Prediksi Varians)")
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(y=res_ngarch.conditional_volatility, mode='lines', name='Volatilitas Kondisional'))
                        fig_vol.update_layout(title_text='Volatilitas Kondisional (NGARCH)', xaxis_rangeslider_visible=True)
                        st.plotly_chart(fig_vol)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat melatih model NGARCH: {e}")
                    st.info("Pastikan residual ARIMA valid dan ordo NGARCH yang dipilih tepat.")
    else:
        st.info("Latih model ARIMA terlebih dahulu di halaman 'Pemodelan ARIMA' untuk mendapatkan residual.")

elif st.session_state['current_page'] == 'pemodelan_arima_ngarch':
    st.markdown('<div class="main-header">Pemodelan ARIMA-NGARCH Terintegrasi</div>', unsafe_allow_html=True)
    st.write("Menggabungkan hasil dari model ARIMA dan NGARCH untuk prediksi nilai tukar dan volatilitas.")

    if 'model_arima_fit' in st.session_state and 'model_ngarch_fit' in st.session_state:
        st.success("Kedua model (ARIMA dan NGARCH) telah dilatih!")
        st.write("Sekarang Anda dapat melihat bagaimana mereka terintegrasi untuk prediksi.")

        # Ini adalah tempat di mana Anda akan mengelola prediksi gabungan.
        # Namun, prediksi akan dilakukan di halaman 'PREDIKSI'.
        st.info("Lanjutkan ke halaman 'PREDIKSI' untuk melihat hasil dan evaluasi model ARIMA-NGARCH.")
    else:
        st.warning("Pastikan Anda telah melatih model ARIMA dan NGARCH di halaman sebelumnya.")


elif st.session_state['current_page'] == 'prediksi':
    st.markdown('<div class="main-header">Prediksi Nilai Tukar dan Volatilitas</div>', unsafe_allow_html=True)
    st.write("Lihat hasil prediksi nilai tukar dari model ARIMA dan prediksi volatilitas dari model NGARCH.")

    if 'model_arima_fit' in st.session_state and 'model_ngarch_fit' in st.session_state and \
       'train_data_returns' in st.session_state and 'test_data_returns' in st.session_state and \
       'original_prices_for_reconstruction' in st.session_state:

        model_arima_fit = st.session_state['model_arima_fit']
        model_ngarch_fit = st.session_state['model_ngarch_fit']
        train_data_returns = st.session_state['train_data_returns']
        test_data_returns = st.session_state['test_data_returns']
        original_prices = st.session_state['original_prices_for_reconstruction']

        st.subheader("Konfigurasi Prediksi")
        num_forecast_steps = st.number_input("Jumlah langkah prediksi ke depan:", min_value=1, max_value=30, value=5, key="num_forecast_steps")

        if st.button("Lakukan Prediksi Gabungan", key="run_combined_prediction_button"):
            try:
                with st.spinner("Melakukan prediksi ARIMA dan NGARCH..."):
                    # --- Prediksi ARIMA (Mean Equation) ---
                    # Lakukan prediksi pada data pengujian
                    # Start index untuk prediksi: setelah data pelatihan
                    # End index untuk prediksi: akhir data pengujian
                    start_idx = len(train_data_returns)
                    end_idx = len(train_data_returns) + len(test_data_returns) - 1
                    
                    # Prediksi return dari model ARIMA
                    # Prediksi future returns dari model ARIMA
                    arima_forecast_returns = model_arima_fit.predict(start=start_idx, end=end_idx, typ='levels') # typ='levels' akan memberi prediksi dalam skala data input
                    
                    # Untuk memprediksi langkah ke depan (out-of-sample)
                    forecast_out_of_sample = model_arima_fit.predict(start=len(original_prices), end=len(original_prices) + num_forecast_steps -1, typ='levels')

                    # --- Rekonstruksi Harga Asli dari Prediksi Return ---
                    # Ini adalah bagian tricky: mengubah prediksi return kembali ke harga asli
                    # Jika menggunakan log return: P_t = P_{t-1} * exp(return_t)
                    # Jika menggunakan simple return: P_t = P_{t-1} * (1 + return_t)

                    # Ambil harga terakhir dari data pelatihan
                    last_train_price = original_prices.iloc[len(train_data_returns)-1] # Harga sebelum dimulainya data test

                    # Rekonstruksi harga untuk data prediksi
                    predicted_prices = [last_train_price] # Mulai dengan harga terakhir pelatihan

                    # Asumsikan return_type adalah 'Log Return' atau 'Simple Return'
                    return_type = st.session_state.get('return_type', 'Log Return') # Ambil dari session state, default Log Return

                    for r in arima_forecast_returns.values:
                        if return_type == "Log Return":
                            next_price = predicted_prices[-1] * np.exp(r)
                        else: # Simple Return
                            next_price = predicted_prices[-1] * (1 + r)
                        predicted_prices.append(next_price)
                    
                    # Hapus elemen pertama karena itu adalah harga terakhir dari train data
                    predicted_prices = np.array(predicted_prices[1:])

                    # Buat Series untuk prediksi harga
                    # Indeks prediksi harus sesuai dengan indeks data uji atau indeks masa depan
                    predicted_prices_series = pd.Series(predicted_prices, index=test_data_returns.index)


                    # --- Prediksi Volatilitas (NGARCH) ---
                    # Prediksi varians/volatilitas kondisional dari model NGARCH
                    # Asumsikan Anda ingin memprediksi volatilitas untuk periode yang sama dengan prediksi harga
                    # Untuk NGARCH, Anda memprediksi varians bersyarat ke depan.
                    # Model NGARCH dilatih pada residual ARIMA. Untuk prediksi ke depan,
                    # kita membutuhkan residual yang diproyeksikan (biasanya 0) dan kemudian
                    # varians bersyarat ke depan.
                    
                    # Menggunakan metode forecast() dari model arch
                    # forecast_res = model_ngarch_fit.forecast(horizon=len(test_data_returns), start=model_ngarch_fit.last_obs)
                    # forecast_variance = forecast_res.variance.iloc[-1]
                    # forecast_volatility = np.sqrt(forecast_variance)
                    
                    # Untuk out-of-sample forecast volatilitas
                    # Perhatikan bahwa ramalan volatilitas mungkin sangat tergantung pada data terakhir
                    forecast_res_ngarch = model_ngarch_fit.forecast(horizon=len(test_data_returns) + num_forecast_steps) # prediksi untuk test set + future
                    # forecast_variance = forecast_res_ngarch.variance.iloc[-1] # ini hanya baris terakhir dari ramalan
                    # Varians bersyarat untuk seluruh horison
                    conditional_variance_forecast = forecast_res_ngarch.variance.values[-1, :] # Ambil baris terakhir dari ramalan varians
                    conditional_volatility_forecast = np.sqrt(conditional_variance_forecast)

                    # Buat Series untuk prediksi volatilitas
                    # Indeks untuk prediksi volatilitas
                    # Ini sedikit tricky karena indeks forecast() dari arch_model berbeda
                    # Jika Anda memprediksi horizon h, itu akan memberikan varians untuk h langkah ke depan
                    # mulai dari observasi terakhir yang digunakan untuk fitting model
                    # Kita bisa membuat indeks dummy untuk ini.
                    volatility_forecast_index = pd.RangeIndex(start=len(original_prices) - len(test_data_returns) +1 , stop=len(original_prices) + num_forecast_steps)
                    # Ini harus disesuaikan agar indeks match dengan harga asli
                    
                    # Cara yang lebih akurat untuk memetakan volatilitas:
                    # Ambil indeks data pengujian
                    test_volatility_index = test_data_returns.index
                    # Kemudian indeks untuk future (out-of-sample)
                    future_index = pd.RangeIndex(start=test_volatility_index.max() + 1, stop=test_volatility_index.max() + 1 + num_forecast_steps)
                    full_forecast_vol_index = test_volatility_index.append(future_index)

                    # Pastikan panjang array conditional_volatility_forecast sesuai
                    if len(conditional_volatility_forecast) < len(full_forecast_vol_index):
                        st.warning("Panjang prediksi volatilitas tidak sesuai dengan horison yang diminta. Mungkin ada masalah dalam forecast NGARCH.")
                        # Pad with NaN atau sesuaikan logika
                        conditional_volatility_forecast_padded = np.pad(conditional_volatility_forecast, (0, len(full_forecast_vol_index) - len(conditional_volatility_forecast)), 'constant', constant_values=np.nan)
                        volatility_forecast_series = pd.Series(conditional_volatility_forecast_padded, index=full_forecast_vol_index)
                    else:
                         volatility_forecast_series = pd.Series(conditional_volatility_forecast[:len(full_forecast_vol_index)], index=full_forecast_vol_index)

                    st.success("Prediksi berhasil dilakukan!")

                    # --- Visualisasi Hasil Prediksi Nilai Tukar ---
                    st.subheader("Prediksi Nilai Tukar (ARIMA)")
                    fig_price = go.Figure()
                    # Plot data harga asli
                    fig_price.add_trace(go.Scatter(x=original_prices.index, y=original_prices['Value'], mode='lines', name='Harga Aktual'))
                    # Plot prediksi ARIMA pada data uji
                    # Perhatikan: Indeks predicted_prices_series harus cocok dengan original_prices untuk plotting yang benar
                    fig_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Test)', line=dict(color='red', dash='dash')))
                    # Plot prediksi future out-of-sample
                    # Rekonstruksi out-of-sample future price
                    last_actual_price = original_prices.iloc[-1]['Value']
                    future_predicted_prices = [last_actual_price]
                    for r_future in forecast_out_of_sample.values:
                        if return_type == "Log Return":
                            next_future_price = future_predicted_prices[-1] * np.exp(r_future)
                        else: # Simple Return
                            next_future_price = future_predicted_prices[-1] * (1 + r_future)
                        future_predicted_prices.append(next_future_price)
                    future_predicted_prices = np.array(future_predicted_prices[1:])
                    future_price_index = pd.RangeIndex(start=original_prices.index.max() + 1, stop=original_prices.index.max() + 1 + num_forecast_steps)
                    future_predicted_prices_series = pd.Series(future_predicted_prices, index=future_price_index)


                    fig_price.add_trace(go.Scatter(x=future_predicted_prices_series.index, y=future_predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Future)', line=dict(color='green', dash='dot')))

                    fig_price.update_layout(title_text='Prediksi Nilai Tukar ARIMA', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_price)

                    # --- Visualisasi Hasil Prediksi Volatilitas ---
                    st.subheader("Prediksi Volatilitas (NGARCH)")
                    fig_volatility = go.Figure()
                    # Plot historical conditional volatility dari model_ngarch_fit
                    fig_volatility.add_trace(go.Scatter(y=model_ngarch_fit.conditional_volatility, mode='lines', name='Volatilitas Aktual (Historis)', line=dict(color='blue')))
                    # Plot prediksi volatilitas
                    fig_volatility.add_trace(go.Scatter(x=volatility_forecast_series.index, y=volatility_forecast_series.values, mode='lines', name='Prediksi Volatilitas (NGARCH)', line=dict(color='purple', dash='dot')))
                    fig_volatility.update_layout(title_text='Prediksi Volatilitas NGARCH', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_volatility)

                    # --- Evaluasi Metrik (untuk data uji) ---
                    st.subheader("Metrik Evaluasi")
                    # Metrik untuk prediksi nilai tukar
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    # Perlu memastikan ukuran original_prices['Value'][start_idx:end_idx+1] sesuai dengan predicted_prices_series
                    actual_test_prices = original_prices['Value'].iloc[start_idx:end_idx+1]
                    
                    if len(actual_test_prices) == len(predicted_prices_series):
                        rmse_price = np.sqrt(mean_squared_error(actual_test_prices, predicted_prices_series))
                        mae_price = mean_absolute_error(actual_test_prices, predicted_prices_series)
                        st.write(f"**Prediksi Nilai Tukar:**")
                        st.write(f"RMSE: {rmse_price:.4f}")
                        st.write(f"MAE: {mae_price:.4f}")
                    else:
                        st.warning("Ukuran data aktual dan prediksi tidak cocok untuk evaluasi harga.")

                    # Metrik untuk volatilitas (lebih kompleks, seringkali menggunakan Loss Function khusus)
                    st.write("**Prediksi Volatilitas:**")
                    st.info("Evaluasi volatilitas lebih kompleks. Metrik umum seperti RMSE atau MAE pada volatilitas yang diprediksi dibandingkan dengan volatilitas aktual mungkin tidak selalu relevan karena volatilitas aktual tidak dapat diamati secara langsung. Seringkali menggunakan proxy seperti return kuadrat atau korelasi.")
                    # Untuk evaluasi, Anda bisa membandingkan conditional_volatility_forecast dengan return kuadrat dari test_data_returns
                    # Namun, itu di luar cakupan contoh sederhana ini.

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi gabungan: {e}")
                st.info("Pastikan semua model telah dilatih dan data tersedia.")
    else:
        st.info("Harap pastikan semua langkah sebelumnya (Input Data, Data Preprocessing, Data Splitting, Pemodelan ARIMA, Pemodelan NGARCH) telah selesai.")

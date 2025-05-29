import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Impor model yang relevan
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

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

# --- Custom CSS untuk Tampilan (Ubah Warna dan Hilangkan Ikon +) ---
st.markdown("""
    <style>
        /* Mengubah warna latar belakang sidebar */
        .css-1d3f8aq.e1fqkh3o1 {
            background-color: #f0f2f6; /* Abu-abu sangat terang */
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
            border: 1px solid #d4d7dc; /* Border abu-abu muda */
            background-color: #ffffff; /* Putih */
            color: #333;
            padding: 0.75rem 1rem;
            font-size: 1rem;
            text-align: left;
            margin-bottom: 0.2rem;
            transition: background-color 0.3s, color 0.3s;
        }
        .stButton>button:hover {
            background-color: #e0e6ed; /* Abu-abu lebih gelap saat hover */
            color: #1a1a1a;
        }
        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(90, 150, 250, 0.25); /* Biru sedikit gelap untuk fokus */
        }
        /* Styling untuk tombol aktif (klik) */
        .stButton>button:active {
            background-color: #a4c6f1; /* Biru sedang saat aktif */
        }
        /* Styling untuk tombol navigasi sidebar yang sedang aktif */
        .stButton button[data-testid^="stSidebarNavButton"]:focus:not(:active) {
            background-color: #dbe9fc !important; /* Biru muda pucat untuk yang aktif */
            font-weight: bold;
            color: #0056b3; /* Biru gelap untuk teks aktif */
        }
        
        /* Header utama aplikasi */
        .main-header {
            background-color: #3f72af; /* Biru tua yang elegan */
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.8em;
            margin-bottom: 2rem;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Sedikit bayangan untuk kesan mendalam */
        }
        /* Sub-header bagian */
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
            color: #2c3e50; /* Warna teks gelap (dark blue-grey) */
        }
        /* Card informasi di halaman HOME */
        .info-card {
            background-color: #ffffff;
            border-radius: 0.5rem;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            text-align: center;
            border-left: 5px solid #3f72af; /* Garis biru tua di sisi kiri */
        }
        /* Menghilangkan ikon '+' */
        .info-card .plus-icon {
            display: none; /* Menyembunyikan ikon + */
        }
        /* Gaya teks interpretasi/saran */
        .interpretation-text {
            background-color: #f8f8f8;
            border-left: 5px solid #3f72af; /* Biru tua */
            padding: 1.5rem;
            margin-top: 1rem;
            margin-bottom: 1rem;
            border-radius: 0.5rem;
        }
        /* Gaya untuk daftar panduan penggunaan */
        .guidance-list ul {
            list-style-type: disc; /* Menggunakan bullet point standar */
            padding-left: 20px;
        }
        .guidance-list li {
            margin-bottom: 10px;
            line-height: 1.5;
        }
        .guidance-list b {
            color: #3f72af; /* Warna biru tua untuk teks bold */
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
    "PEMODELAN ARIMA": "pemodelan_arima",
    "PEMODELAN NGARCH": "pemodelan_ngarch",
    "PEMODELAN ARIMA-NGARCH": "pemodelan_arima_ngarch",
    "PREDIKSI": "prediksi",
    "INTERPRETASI & SARAN": "interpretasi_saran",
}

if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'

for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# --- Area Konten Utama Berdasarkan Halaman yang Dipilih ---

if st.session_state['current_page'] == 'home':
    st.markdown('<div class="main-header">Prediksi Data Time Series Univariat <br> Menggunakan Model ARIMA-NGARCH</div>', unsafe_allow_html=True)

    # Ikon '+' dihapus, hanya ada teks
    st.markdown("""
        <div class="info-card">
            <p>Sistem ini dirancang untuk melakukan prediksi nilai tukar mata uang menggunakan model ARIMA dan mengukur volatilitasnya dengan model NGARCH.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">Panduan Penggunaan Sistem</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="guidance-list">
    <ul>
        <li><b>HOME:</b> Halaman utama yang menjelaskan tujuan dan metode prediksi sistem.</li>
        <li><b>INPUT DATA:</b> Unggah data time series nilai tukar mata uang.</li>
        <li><b>DATA PREPROCESSING:</b> Lakukan pembersihan dan transformasi data (misalnya, menghitung return).</li>
        <li><b>STASIONERITAS DATA:</b> Uji stasioneritas data return sebelum model ARIMA dibentuk.</li>
        <li><b>DATA SPLITTING:</b> Pisahkan data menjadi latih dan uji.</li>
        <li><b>PEMODELAN ARIMA:</b> Langkah-langkah untuk membentuk model ARIMA pada data return (untuk prediksi nilai tukar).</li>
        <li><b>PEMODELAN NGARCH:</b> Langkah-langkah untuk membentuk model NGARCH pada residual ARIMA (untuk prediksi volatilitas).</li>
        <li><b>PEMODELAN ARIMA-NGARCH:</b> Integrasi model ARIMA dan NGARCH.</li>
        <li><b>PREDIKSI:</b> Menampilkan hasil prediksi nilai tukar dan volatilitas.</li>
        <li><b>INTERPRETASI & SARAN:</b> Penjelasan hasil model dan rekomendasi.</li>
    </ul>
    </div>
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
            st.error(f"Terjadi kesalahan saat membaca file yang diunggah: {e}")
            return pd.DataFrame()
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
        fig_raw.add_trace(go.Scatter(y=st.session_state['df_currency_raw']['Value'], mode='lines', name='Nilai Tukar', line=dict(color='#5d8aa8'))) # Warna garis lebih lembut
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
                    st.session_state['return_type'] = return_type # Simpan tipe return
                    st.success("Data return berhasil dihitung.")
                    st.write("5 baris pertama data return:")
                    st.dataframe(processed_series.head())

                    st.subheader("Visualisasi Data Return")
                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Scatter(y=processed_series, mode='lines', name='Data Return', line=dict(color='#82c0cc'))) # Warna garis lebih lembut
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
                    st.session_state['is_stationary'] = True
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
            fig_split.add_trace(go.Scatter(x=train_data_returns.index, y=train_data_returns.values, mode='lines', name='Data Pelatihan', line=dict(color='#3f72af'))) # Warna biru tua
            fig_split.add_trace(go.Scatter(x=test_data_returns.index, y=test_data_returns.values, mode='lines', name='Data Pengujian', line=dict(color='#ff7f0e'))) # Warna oranye
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
        p = st.number_input("Ordo AR (p):", min_value=0, max_value=5, value=1, key="arima_p")
        d = st.number_input("Ordo Differencing (d):", min_value=0, max_value=2, value=0, key="arima_d")
        q = st.number_input("Ordo MA (q):", min_value=0, max_value=5, value=1, key="arima_q")

        if st.button("Latih Model ARIMA", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA..."):
                    model_arima = ARIMA(train_data_returns, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.success("Model ARIMA berhasil dilatih!")
                    st.subheader("Ringkasan Model ARIMA")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("Residual Model ARIMA")
                    fig_res = go.Figure()
                    fig_res.add_trace(go.Scatter(y=model_arima_fit.resid, mode='lines', name='Residual ARIMA', line=dict(color='#4c78a8'))) # Warna biru
                    fig_res.update_layout(title_text='Residual Model ARIMA', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_res)

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
        arima_residuals = st.session_state['arima_residuals'].dropna()
        st.write("Residual dari model ARIMA (data untuk model NGARCH):")
        st.dataframe(arima_residuals.head())

        if arima_residuals.empty:
            st.warning("Residual ARIMA kosong atau hanya berisi NaN. Pastikan model ARIMA berhasil dilatih dan menghasilkan residual yang valid.")
        else:
            st.subheader("Pilih Ordo NGARCH (p, o, q)")
            p_garch = st.number_input("Ordo ARCH (p):", min_value=1, max_value=3, value=1, key="ngarch_p")
            o_garch = st.number_input("Ordo Asymmetry (o):", min_value=0, max_value=2, value=1, key="ngarch_o")
            q_garch = st.number_input("Ordo GARCH (q):", min_value=1, max_value=3, value=1, key="ngarch_q")

            if st.button("Latih Model NGARCH", key="train_ngarch_button"):
                try:
                    with st.spinner("Melatih model NGARCH..."):
                        # Menggunakan 'Garch' sebagai vol_model dengan argumen `o` yang menentukan asimetri (GJR-GARCH)
                        model_ngarch = arch_model(arima_residuals, vol='Garch', p=p_garch, o=o_garch, q=q_garch, dist='t')
                        res_ngarch = model_ngarch.fit(disp='off')

                        st.session_state['model_ngarch_fit'] = res_ngarch
                        st.success("Model NGARCH (GJR-GARCH) berhasil dilatih!")
                        st.subheader("Ringkasan Model NGARCH")
                        st.text(res_ngarch.summary().as_text())

                        st.subheader("Volatilitas Kondisional (Prediksi Varians)")
                        fig_vol = go.Figure()
                        fig_vol.add_trace(go.Scatter(y=res_ngarch.conditional_volatility, mode='lines', name='Volatilitas Kondisional', line=dict(color='#2ca02c'))) # Warna hijau
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
        return_type = st.session_state.get('return_type', 'Log Return')

        st.subheader("Konfigurasi Prediksi")
        num_forecast_steps = st.number_input("Jumlah langkah prediksi ke depan:", min_value=1, max_value=30, value=5, key="num_forecast_steps")

        if st.button("Lakukan Prediksi Gabungan", key="run_combined_prediction_button"):
            try:
                with st.spinner("Melakukan prediksi ARIMA dan NGARCH..."):
                    # --- Prediksi ARIMA (Mean Equation) ---
                    # Perbaikan: Mengambil index terakhir dari data training untuk memulai prediksi
                    # Ini lebih robust jika index bukan numerik sederhana
                    start_pred_idx = len(train_data_returns)
                    end_pred_idx = len(train_data_returns) + len(test_data_returns) - 1

                    arima_forecast_returns_test = model_arima_fit.predict(start=start_pred_idx, end=end_pred_idx, typ='levels')
                    
                    # Prediksi Out-of-Sample (Future)
                    # Ambil indeks terakhir dari data harga asli untuk kelanjutan
                    future_start_idx_for_predict = len(original_prices)
                    future_end_idx_for_predict = len(original_prices) + num_forecast_steps - 1
                    forecast_out_of_sample_returns = model_arima_fit.predict(start=future_start_idx_for_predict, end=future_end_idx_for_predict, typ='levels')

                    # --- Rekonstruksi Harga Asli dari Prediksi Return ---
                    # Rekonstruksi untuk data uji
                    last_train_price = original_prices.iloc[len(train_data_returns)-1]['Value']
                    predicted_prices_test = [last_train_price]

                    for r in arima_forecast_returns_test.values:
                        if return_type == "Log Return":
                            next_price = predicted_prices_test[-1] * np.exp(r)
                        else:
                            next_price = predicted_prices_test[-1] * (1 + r)
                        predicted_prices_test.append(next_price)
                    predicted_prices_test = np.array(predicted_prices_test[1:])
                    predicted_prices_series = pd.Series(predicted_prices_test, index=test_data_returns.index)

                    # Rekonstruksi untuk future prediction
                    last_actual_price_full_data = original_prices.iloc[-1]['Value']
                    future_predicted_prices_list = [last_actual_price_full_data]
                    
                    # Buat indeks untuk prediksi masa depan
                    # Asumsi indeks original_prices adalah numerik atau DateTimeIndex yang berurutan
                    if isinstance(original_prices.index, pd.DatetimeIndex):
                        last_date = original_prices.index.max()
                        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=num_forecast_steps, freq='D') # Asumsi harian
                        future_price_index = future_dates
                    else:
                        future_price_index = pd.RangeIndex(start=original_prices.index.max() + 1, stop=original_prices.index.max() + 1 + num_forecast_steps)

                    for r_future in forecast_out_of_sample_returns.values:
                        if return_type == "Log Return":
                            next_future_price = future_predicted_prices_list[-1] * np.exp(r_future)
                        else:
                            next_future_price = future_predicted_prices_list[-1] * (1 + r_future)
                        future_predicted_prices_list.append(next_future_price)
                    future_predicted_prices_series = pd.Series(future_predicted_prices_list[1:], index=future_price_index)


                    # --- Prediksi Volatilitas (NGARCH) ---
                    # Prediksi volatilitas untuk keseluruhan horison test set + future
                    # arch_model.forecast mengembalikan objek, ambil nilai varians dari sana
                    forecast_res_ngarch = model_ngarch_fit.forecast(horizon=len(test_data_returns) + num_forecast_steps, method='simulation', simulations=1000) # Bisa juga 'analytic'

                    # Ambil mean dari simulasi varians
                    conditional_variance_forecast_mean = forecast_res_ngarch.variance.mean.values[-1, :]
                    conditional_volatility_forecast = np.sqrt(conditional_variance_forecast_mean)

                    # Buat index yang sesuai untuk prediksi volatilitas
                    # Ini harus dimulai dari indeks setelah volatilitas kondisional terakhir dari data latih
                    last_vol_index = model_ngarch_fit.conditional_volatility.index.max()
                    if isinstance(last_vol_index, pd.Timestamp):
                        full_forecast_vol_index = pd.date_range(start=last_vol_index + pd.Timedelta(days=1), periods=len(test_data_returns) + num_forecast_steps, freq='D')
                    else:
                        full_forecast_vol_index = pd.RangeIndex(start=last_vol_index + 1, stop=last_vol_index + 1 + len(test_data_returns) + num_forecast_steps)

                    volatility_forecast_series = pd.Series(conditional_volatility_forecast[:len(full_forecast_vol_index)], index=full_forecast_vol_index)

                    st.success("Prediksi berhasil dilakukan!")

                    # --- Visualisasi Hasil Prediksi Nilai Tukar ---
                    st.subheader("Prediksi Nilai Tukar (ARIMA)")
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(x=original_prices.index, y=original_prices['Value'], mode='lines', name='Harga Aktual', line=dict(color='#3f72af'))) # Biru tua
                    fig_price.add_trace(go.Scatter(x=predicted_prices_series.index, y=predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Test)', line=dict(color='#d62728', dash='dash'))) # Merah
                    fig_price.add_trace(go.Scatter(x=future_predicted_prices_series.index, y=future_predicted_prices_series.values, mode='lines', name='Prediksi ARIMA (Future)', line=dict(color='#2ca02c', dash='dot'))) # Hijau
                    fig_price.update_layout(title_text='Prediksi Nilai Tukar ARIMA', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_price)

                    # --- Visualisasi Hasil Prediksi Volatilitas ---
                    st.subheader("Prediksi Volatilitas (NGARCH)")
                    fig_volatility = go.Figure()
                    fig_volatility.add_trace(go.Scatter(x=model_ngarch_fit.conditional_volatility.index, y=model_ngarch_fit.conditional_volatility, mode='lines', name='Volatilitas Aktual (Historis)', line=dict(color='#8c564b'))) # Coklat
                    fig_volatility.add_trace(go.Scatter(x=volatility_forecast_series.index, y=volatility_forecast_series.values, mode='lines', name='Prediksi Volatilitas (NGARCH)', line=dict(color='#9467bd', dash='dot'))) # Ungu
                    fig_volatility.update_layout(title_text='Prediksi Volatilitas NGARCH', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_volatility)

                    # --- Evaluasi Metrik (untuk data uji) ---
                    from sklearn.metrics import mean_squared_error, mean_absolute_error
                    actual_test_prices = original_prices['Value'].iloc[start_pred_idx:end_pred_idx+1]

                    st.subheader("Metrik Evaluasi")
                    if len(actual_test_prices) == len(predicted_prices_series):
                        rmse_price = np.sqrt(mean_squared_error(actual_test_prices, predicted_prices_series))
                        mae_price = mean_absolute_error(actual_test_prices, predicted_prices_series)
                        st.write(f"**Prediksi Nilai Tukar (pada data uji):**")
                        st.write(f"RMSE: {rmse_price:.4f}")
                        st.write(f"MAE: {mae_price:.4f}")
                        st.session_state['rmse_price'] = rmse_price
                        st.session_state['mae_price'] = mae_price
                    else:
                        st.warning("Ukuran data aktual dan prediksi tidak cocok untuk evaluasi harga pada data uji.")

                    st.write("**Prediksi Volatilitas:**")
                    st.info("Evaluasi volatilitas lebih kompleks. Metrik umum seperti RMSE atau MAE pada volatilitas yang diprediksi dibandingkan dengan volatilitas aktual mungkin tidak selalu relevan karena volatilitas aktual tidak dapat diamati secara langsung. Seringkali menggunakan proxy seperti return kuadrat atau korelasi.")
                    # Simpan data penting untuk interpretasi
                    st.session_state['last_forecast_price'] = future_predicted_prices_series.iloc[-1] if not future_predicted_prices_series.empty else None
                    st.session_state['last_forecast_volatility'] = volatility_forecast_series.iloc[-1] if not volatility_forecast_series.empty else None

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi gabungan: {e}")
                st.info("Pastikan semua model telah dilatih dan data tersedia.")
    else:
        st.info("Harap pastikan semua langkah sebelumnya (Input Data, Data Preprocessing, Data Splitting, Pemodelan ARIMA, Pemodelan NGARCH) telah selesai.")


elif st.session_state['current_page'] == 'interpretasi_saran':
    st.markdown('<div class="main-header">Interpretasi Hasil dan Saran</div>', unsafe_allow_html=True)
    st.write("Bagian ini memberikan interpretasi terhadap hasil pemodelan ARIMA-NGARCH dan beberapa saran praktis.")

    if 'model_arima_fit' in st.session_state and 'model_ngarch_fit' in st.session_state:
        st.subheader("Interpretasi Model ARIMA-NGARCH")
        st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
        st.write("""
        Model ARIMA (AutoRegressive Integrated Moving Average) digunakan untuk memodelkan komponen mean dari data return nilai tukar.
        Ini membantu kita memprediksi arah pergerakan nilai tukar di masa depan.
        """)
        if 'model_arima_fit' in st.session_state:
            st.write(f"- Ordo ARIMA yang digunakan adalah: {st.session_state['model_arima_fit'].model.order}")
            st.write("Parameter-parameter ini menentukan berapa banyak observasi masa lalu yang digunakan untuk memprediksi nilai saat ini (AR), seberapa banyak differencing yang dilakukan (I), dan seberapa banyak kesalahan prediksi masa lalu yang digunakan (MA).")
        st.write("""
        Model NGARCH (Non-linear Generalized Autoregressive Conditional Heteroskedasticity), yang diimplementasikan di sini sebagai GJR-GARCH,
        digunakan untuk memodelkan volatilitas (varians bersyarat) dari residual model ARIMA.
        """)
        if 'model_ngarch_fit' in st.session_state:
            st.write(f"- Ordo NGARCH (GJR-GARCH) yang digunakan: p={st.session_state['model_ngarch_fit'].model.p}, o={st.session_state['model_ngarch_fit'].model.o}, q={st.session_state['model_ngarch_fit'].model.q}")
            st.write("Model NGARCH menangkap volatilitas yang berubah seiring waktu dan juga efek asimetris (misalnya, berita buruk mungkin memiliki dampak yang lebih besar pada volatilitas dibandingkan berita baik).")
        st.write('</div>', unsafe_allow_html=True)

        st.subheader("Evaluasi Kinerja Model")
        st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
        if 'rmse_price' in st.session_state and 'mae_price' in st.session_state:
            st.write(f"**Prediksi Nilai Tukar:**")
            st.write(f"- RMSE (Root Mean Squared Error): {st.session_state['rmse_price']:.4f}")
            st.write(f"- MAE (Mean Absolute Error): {st.session_state['mae_price']:.4f}")
            st.write("Nilai RMSE dan MAE yang lebih rendah menunjukkan akurasi prediksi nilai tukar yang lebih baik.")
        else:
            st.info("Silakan jalankan prediksi terlebih dahulu di halaman 'PREDIKSI' untuk melihat metrik evaluasi.")
        st.write("""
        **Volatilitas:**
        Evaluasi akurasi prediksi volatilitas lebih kompleks karena volatilitas 'aktual' tidak dapat langsung diamati.
        Namun, model NGARCH memberikan perkiraan volatilitas bersyarat yang dapat digunakan sebagai indikator risiko.
        Semakin tinggi volatilitas yang diprediksi, semakin besar fluktuasi nilai tukar yang diantisipasi.
        """)
        st.write('</div>', unsafe_allow_html=True)


        st.subheader("Kesimpulan dan Saran")
        st.markdown('<div class="interpretation-text">', unsafe_allow_html=True)
        st.write("""
        Model ARIMA-NGARCH adalah alat yang kuat untuk memprediksi nilai tukar dan volatilitasnya.
        Prediksi nilai tukar membantu dalam pengambilan keputusan investasi atau transaksi di masa depan,
        sementai prediksi volatilitas memberikan wawasan tentang tingkat risiko yang mungkin terjadi.
        """)

        if st.session_state.get('last_forecast_price') is not None and st.session_state.get('last_forecast_volatility') is not None:
            st.write(f"**Berdasarkan prediksi terbaru:**")
            st.write(f"- Nilai tukar yang diprediksi untuk periode selanjutnya: **{st.session_state['last_forecast_price']:.4f}**")
            st.write(f"- Prediksi volatilitas untuk periode selanjutnya: **{st.session_state['last_forecast_volatility']:.4f}** (semakin tinggi, semakin besar fluktuasi yang diharapkan)")
            st.write("*(Catatan: Prediksi ini adalah berdasarkan data dan model yang dilatih, selalu perbarui model dengan data terbaru untuk hasil yang relevan.)*")
        
        st.write("""
        **Saran:**
        <ol>
            <li><b>Validasi Ulang Model:</b> Model time series sensitif terhadap perubahan kondisi pasar. Lakukan validasi dan pelatihan ulang model secara berkala dengan data terbaru.</li>
            <li><b>Perbandingan Model:</b> Pertimbangkan untuk membandingkan kinerja ARIMA-NGARCH dengan model lain (misalnya, GARCH murni, E-GARCH, atau model pembelajaran mesin) untuk menemukan yang paling sesuai dengan karakteristik data Anda.</li>
            <li><b>Analisis Residual:</b> Selalu periksa residual model (terutama ARIMA) untuk memastikan tidak ada pola yang tersisa, yang mengindikasikan bahwa model belum menangkap semua informasi.</li>
            <li><b>Wawasan Domain:</b> Gabungkan hasil prediksi dengan wawasan ekonomi, geopolitik, dan sentimen pasar yang relevan untuk pengambilan keputusan yang lebih komprehensif.</li>
            <li><b>Manajemen Risiko:</b> Gunakan prediksi volatilitas sebagai indikator risiko untuk menyesuaikan strategi investasi atau hedging Anda.</li>
        </ol>
        """)
        st.write('</div>', unsafe_allow_html=True)

    else:
        st.info("Silakan lengkapi langkah pemodelan ARIMA dan NGARCH serta prediksi untuk melihat interpretasi dan saran.")

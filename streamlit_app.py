import streamlit as st
import pandas as pd
import math
from pathlib import Path

# --- Konfigurasi Halaman (Hanya dipanggil sekali di awal) ---
st.set_page_config(
    page_title='Prediksi ARIMA-NGARCH dan ANFIS',
    page_icon='📈',  # Anda bisa menggunakan emoji atau URL yang valid
    layout="wide" # Memastikan tata letak lebar secara default
)

# --- Fungsi Pembaca Data (dengan caching) ---
# Menggunakan st.cache_data untuk caching data yang diunggah atau default.
# TTL (Time To Live) 86400 detik = 1 hari.
@st.cache_data(ttl=86400)
def load_data(uploaded_file=None, default_filename='data/default.csv'):
    """
    Membaca data dari objek file yang diunggah atau dari file default lokal.
    """
    if uploaded_file is not None:
        try:
            # Pandas dapat membaca langsung dari objek BytesIO yang diberikan oleh Streamlit's UploadedFile
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file yang diunggah: {e}")
            return pd.DataFrame() # Mengembalikan DataFrame kosong jika ada error
    else:
        # Coba membaca dari file default jika ada
        # Pastikan Anda memiliki folder 'data' dan 'default.csv' di root proyek Anda
        path = Path(__file__).parent / default_filename
        if path.exists():
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.warning(f"Tidak dapat membaca file default '{default_filename}': {e}")
                return pd.DataFrame()
        else:
            st.warning(f"File default '{default_filename}' tidak ditemukan.")
            return pd.DataFrame()

# --- Custom CSS untuk Tampilan ---
st.markdown("""
    <style>
        /* Mengubah warna latar belakang sidebar */
        .css-1d3f8aq.e1fqkh3o1 { /* Ini class untuk sidebar di Streamlit 1.x */
            background-color: #f0f2f6; /* Abu-abu muda untuk latar belakang sidebar */
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Mengatur padding untuk konten utama */
        .css-1v0mbdj.e1fqkh3o0 { /* Ini class untuk main content di Streamlit 1.x */
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 5rem; /* Sesuaikan sesuai kebutuhan */
            padding-right: 5rem; /* Sesuaikan sesuai kebutuhan */
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
            margin-bottom: 0.2rem; /* Sedikit spasi antar tombol */
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

        /* --- Styling untuk Tombol Navigasi Aktif --- */
        /* Ini adalah trik CSS untuk mencoba memberikan efek "aktif" visual
           pada tombol yang dipilih. Streamlit sering mengubah class internal,
           jadi ini mungkin perlu disesuaikan jika Streamlit versi Anda berubah.
           Ini menargetkan tombol yang sedang difokuskan (setelah diklik) dan
           tidak dalam keadaan 'active' (yaitu, mouse sudah dilepas).
           Ini adalah heuristik, bukan jaminan 100% karena kontrol Streamlit.
        */
        .stButton button[data-testid^="stSidebarNavButton"]:focus:not(:active) {
            background-color: #d0d2d6 !important; /* Contoh warna aktif */
            font-weight: bold;
        }
        /* Untuk memastikan tombol yang sedang terpilih tetap 'aktif' walaupun fokus hilang */
        /* Ini akan jauh lebih rumit tanpa JavaScript atau komponen kustom.
           Alternatif: Menambahkan class CSS melalui JavaScript injeksi.
           Untuk saat ini, mari fokus pada fungsionalitas dan CSS yang lebih sederhana.
        */


        /* Styling untuk header utama */
        .main-header {
            background-color: #c94f71; /* Contoh: Merah muda/merah tua dari gambar */
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            text-align: center;
            font-size: 1.8em;
            margin-bottom: 2rem;
        }
        /* Styling untuk header bagian */
        .section-header {
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        /* Styling untuk kartu informasi (kotak dengan ikon plus) */
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
            color: #c94f71; /* Menyesuaikan warna header */
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Menu ---
st.sidebar.markdown("#### MENU NAVIGASI")

# Mendefinisikan item menu dan konten halaman yang sesuai
menu_items = {
    "HOME": "home",
    "INPUT DATA": "input_data",
    "DATA PREPROCESSING": "data_preprocessing",
    "STASIONERITAS DATA": "stasioneritas_data",
    "DATA SPLITTING": "data_splitting",
    "PEMODELAN ARIMA": "pemodelan_arima",
    "PEMODELAN ANFIS ABC": "pemodelan_anfis_abc",
    "PEMODELAN ARIMA-ANFIS ABC": "pemodelan_arima_anfis_abc",
    "PREDIKSI": "prediksi",
    "ARIMA-NGARCH": "arima_ngarch"
}

# Menggunakan session state untuk mengelola halaman aktif
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home' # Halaman default

# Loop untuk membuat tombol di sidebar
for item, key in menu_items.items():
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# --- Area Konten Utama Berdasarkan Halaman yang Dipilih ---

if st.session_state['current_page'] == 'home':
    st.markdown('<div class="main-header">Prediksi Data Time Series Univariat <br> Menggunakan Model ARIMA-ANFIS dengan Optimasi ABC</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <div class="plus-icon">+</div>
            <p>Sistem ini dirancang untuk melakukan prediksi pada <b>data univariat</b> dengan menggunakan model yang dibangun berdasarkan pola dari data historis, sehingga dapat memberikan estimasi yang lebih akurat dan relevan terhadap kondisi nyata. Hasil prediksi ini dapat digunakan untuk membantu pengambilan keputusan dan perencanaan secara lebih efisien.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<h3 class="section-header">Panduan Penggunaan Sistem</h3>', unsafe_allow_html=True)
    st.markdown("""
    <ul>
        <li><b>HOME:</b> Halaman utama yang menjelaskan tujuan dan metode prediksi sistem.</li>
        <li><b>INPUT DATA:</b> Unggah data time series yang akan digunakan dalam pemodelan.</li>
        <li><b>DATA PREPROCESSING:</b> Lakukan pembersihan data.</li>
        <li><b>STASIONERITAS DATA:</b> Uji stasioneritas sebelum model ARIMA dibentuk.</li>
        <li><b>DATA SPLITTING:</b> Pisahkan data menjadi latih dan uji.</li>
        <li><b>PEMODELAN ARIMA:</b> Langkah-langkah untuk membentuk model ARIMA.</li>
        <li><b>PEMODELAN ANFIS ABC:</b> Langkah-langkah untuk membentuk model ANFIS dengan optimasi ABC.</li>
        <li><b>PEMODELAN ARIMA-ANFIS ABC:</b> Integrasi model ARIMA dan ANFIS ABC.</li>
        <li><b>PREDIKSI:</b> Menampilkan hasil prediksi dari model yang telah dibuat.</li>
        <li><b>ARIMA-NGARCH:</b> Bagian khusus untuk prediksi menggunakan model ARIMA-NGARCH dan volatilitas mata uang.</li>
    </ul>
    """, unsafe_allow_html=True)

elif st.session_state['current_page'] == 'input_data':
    st.markdown('<div class="main-header">Input Data</div>', unsafe_allow_html=True)
    st.write("Di sinilah Anda dapat mengunggah berbagai jenis data time series untuk aplikasi Anda. Data yang diunggah akan disimpan dalam sesi untuk digunakan di halaman lain.")

    uploaded_file_input_data_page = st.file_uploader("Pilih file data (CSV, Excel, dll.)", type=["csv", "xlsx", "txt"], key="input_data_uploader")

    if uploaded_file_input_data_page is not None:
        try:
            # Memanggil fungsi load_data yang sudah dicache
            df_general = load_data(uploaded_file=uploaded_file_input_data_page)

            if not df_general.empty:
                st.success("File berhasil diunggah dan dibaca!")
                st.write("5 baris pertama dari data Anda:")
                st.dataframe(df_general.head())
                st.session_state['df_general'] = df_general # Simpan ke session state

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.warning("Pastikan file yang diunggah adalah file yang valid dan diformat dengan benar.")
    elif 'df_general' not in st.session_state or st.session_state['df_general'].empty:
        # Jika tidak ada file diunggah di halaman ini, dan tidak ada data di session state,
        # coba muat data default.
        st.info("Tidak ada file yang diunggah. Mencoba memuat data default 'data/default.csv'.")
        df_general_default = load_data(uploaded_file=None, default_filename='data/default.csv')
        if not df_general_default.empty:
            st.success("Data default berhasil dimuat.")
            st.dataframe(df_general_default.head())
            st.session_state['df_general'] = df_general_default
        else:
            st.warning("Tidak ada data yang dimuat. Silakan unggah file atau pastikan 'data/default.csv' ada.")
    else:
        st.write("Data yang sudah diunggah sebelumnya:")
        st.dataframe(st.session_state['df_general'].head())


elif st.session_state['current_page'] == 'arima_ngarch':
    st.markdown('<div class="main-header">💹 Prediksi ARIMA-NGARCH dengan Model Volatilitas Mata Uang</div>', unsafe_allow_html=True)
    st.markdown("""
        Unggah data time series mata uang Anda untuk melakukan analisis dan prediksi menggunakan model ARIMA dan NGARCH.
        Dashboard ini memudahkan visualisasi hasil prediksi dan evaluasi model volatilitas.
    """)
    st.subheader("Data untuk Analisis ARIMA-NGARCH")

    # Coba gunakan data yang sudah diunggah di halaman 'Input Data'
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Menggunakan data yang sudah dimuat dari halaman 'Input Data':")
        df_currency = st.session_state['df_general']
        st.dataframe(df_currency.head())

        # Anda bisa menambahkan logika di sini untuk memilih kolom dari df_currency
        # Misalnya: selected_column = st.selectbox("Pilih kolom mata uang:", df_currency.columns)
        # Atau melakukan prediksi langsung jika strukturnya sudah diketahui.

    else:
        st.info("Tidak ada data yang dimuat. Silakan unggah data di halaman 'Input Data' atau di sini.")
        # Opsi unggah file khusus untuk halaman ARIMA-NGARCH jika ingin terpisah
        uploaded_file_arima_ngarch_page = st.file_uploader("Atau, unggah file CSV data mata uang di sini:", type=["csv"], key="arima_ngarch_uploader")
        if uploaded_file_arima_ngarch_page is not None:
            df_currency = load_data(uploaded_file=uploaded_file_arima_ngarch_page)
            if not df_currency.empty:
                st.success("File berhasil diunggah dan dibaca di halaman ARIMA-NGARCH!")
                st.dataframe(df_currency.head())
                st.session_state['df_currency_specific'] = df_currency # Simpan di session state yang berbeda
            else:
                st.warning("Gagal memuat data dari file yang diunggah.")
        elif 'df_currency_specific' in st.session_state and not st.session_state['df_currency_specific'].empty:
            st.write("Menggunakan data yang sudah dimuat sebelumnya di halaman ini:")
            df_currency = st.session_state['df_currency_specific']
            st.dataframe(df_currency.head())
        else:
            st.warning("Tidak ada data yang tersedia untuk analisis ARIMA-NGARCH. Silakan unggah data.")


    st.subheader("Hasil Prediksi dan Visualisasi")
    st.info("Area ini akan menampilkan grafik prediksi, volatilitas, dan metrik evaluasi model.")
    # Contoh placeholder:
    if 'df_currency' in st.session_state and not st.session_state['df_currency'].empty:
        st.write("Contoh tampilan data yang diunggah (dari df_general):")
        # Anggap kolom pertama adalah time series
        st.line_chart(st.session_state['df_currency'].iloc[:, 0])
    elif 'df_currency_specific' in st.session_state and not st.session_state['df_currency_specific'].empty:
        st.write("Contoh tampilan data yang diunggah (dari df_currency_specific):")
        st.line_chart(st.session_state['df_currency_specific'].iloc[:, 0])
    else:
        st.info("Unggah data untuk melihat contoh visualisasi.")


elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">Data Preprocessing</div>', unsafe_allow_html=True)
    st.write("Lakukan langkah-langkah pembersihan dan persiapan data di bagian ini, seperti penanganan nilai hilang, normalisasi, dll.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang tersedia untuk preprocessing:")
        st.dataframe(st.session_state['df_general'].head())
        # Tambahkan opsi preprocessing di sini
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'stasioneritas_data':
    st.markdown('<div class="main-header">Stasioneritas Data</div>', unsafe_allow_html=True)
    st.write("Lakukan pengujian stasioneritas data time series (misalnya Uji ADF atau KPSS) di sini.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang akan diuji stasioneritasnya:")
        st.dataframe(st.session_state['df_general'].head())
        st.info("Integrasikan pustaka statistik seperti `statsmodels` untuk uji stasioneritas di sini.")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">Data Splitting</div>', unsafe_allow_html=True)
    st.write("Pisahkan data menjadi set pelatihan dan pengujian untuk evaluasi model.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang akan dibagi:")
        st.dataframe(st.session_state['df_general'].head())
        train_ratio = st.slider("Pilih rasio data pelatihan:", 0.5, 0.9, 0.8, 0.05)
        st.write(f"Rasio pelatihan: {train_ratio*100}%")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'pemodelan_arima':
    st.markdown('<div class="main-header">Pemodelan ARIMA</div>', unsafe_allow_html=True)
    st.write("Bangun dan konfigurasi model ARIMA Anda di bagian ini.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang akan dimodelkan dengan ARIMA:")
        st.dataframe(st.session_state['df_general'].head())
        st.info("Tambahkan parameter ARIMA (p, d, q) dan hasil model di sini.")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'pemodelan_anfis_abc':
    st.markdown('<div class="main-header">Pemodelan ANFIS ABC</div>', unsafe_allow_html=True)
    st.write("Konfigurasi dan latih model ANFIS dengan optimasi Algoritma Koloni Lebah (ABC).")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang akan dimodelkan dengan ANFIS ABC:")
        st.dataframe(st.session_state['df_general'].head())
        st.info("Tambahkan konfigurasi ANFIS dan kontrol ABC di sini.")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'pemodelan_arima_anfis_abc':
    st.markdown('<div class="main-header">Pemodelan ARIMA-ANFIS ABC</div>', unsafe_allow_html=True)
    st.write("Integrasi dan pelatihan model gabungan ARIMA dan ANFIS ABC.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang akan dimodelkan dengan ARIMA-ANFIS ABC:")
        st.dataframe(st.session_state['df_general'].head())
        st.info("Integrasikan hasil dari pemodelan ARIMA dan ANFIS ABC di sini.")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

elif st.session_state['current_page'] == 'prediksi':
    st.markdown('<div class="main-header">Prediksi</div>', unsafe_allow_html=True)
    st.write("Lihat dan evaluasi hasil prediksi dari model yang telah Anda latih.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang digunakan untuk prediksi:")
        st.dataframe(st.session_state['df_general'].head())
        st.info("Tampilkan grafik prediksi dan metrik evaluasi (RMSE, MAE, dll.) di sini.")
    else:
        st.info("Unggah data terlebih dahulu di bagian 'Input Data'.")

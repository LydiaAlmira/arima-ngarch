import streamlit as st
import pandas as pd
import math
from pathlib import Path

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Prediksi ARIMA-NGARCH',
    page_icon='https://example.com/currency-icon.png'  # Ganti dengan URL ikon kamu
)

# Fungsi pembaca data yang fleksibel dan memiliki TTL cache
@st.cache_data(ttl=86400)  # TTL 86400 detik = 1 hari
def load_data(uploaded_file=None, default_filename='data/default.csv'):
    """
    Membaca data dari file upload atau default, dan menyimpan hasilnya di cache
    selama maksimal 1 hari (TTL = 86400 detik).
    """
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        path = Path(__file__).parent / default_filename
        df = pd.read_csv(path)
    return df

# Sidebar untuk upload
st.sidebar.header("Unggah File CSV Anda")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type="csv")

# Panggil fungsi dengan cache
df = load_data(uploaded_file)

# Tampilkan
st.subheader("Data yang Dimuat")
st.dataframe(df)

# -----------------------------------------------------------------------------
# Menggambar halaman utama

# Konfigurasi halaman untuk tata letak yang lebih lebar
st.set_page_config(layout="wide")

# CSS Kustom untuk sidebar dan area konten utama agar menyerupai gambar
st.markdown("""
    <style>
        /* Mengubah warna latar belakang sidebar */
        .css-1d3f8aq.e1fqkh3o1 {
            background-color: #f0f2f6; /* Abu-abu muda untuk latar belakang sidebar */
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        /* Mengatur padding untuk konten utama */
        .css-1v0mbdj.e1fqkh3o0 {
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
        }
        .stButton>button:hover {
            background-color: #e0e2e6;
        }
        .stButton>button:focus {
            outline: none;
            box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
        }
        /* Styling untuk tombol aktif */
        .stButton>button.active {
            background-color: #d0d2d6; /* Sedikit lebih gelap untuk status aktif */
            font-weight: bold;
        }
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

# Menu Sidebar
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
    "ARIMA-NGARCH": "arima_ngarch" # Item baru untuk permintaan Anda
}

# Menggunakan session state untuk mengelola halaman aktif
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home' # Halaman default

for item, key in menu_items.items():
    # Menambahkan class 'active' ke tombol yang sedang aktif
    button_class = "active" if st.session_state['current_page'] == key else ""
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key

# Area Konten Utama
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
        </ul>
    """, unsafe_allow_html=True)

elif st.session_state['current_page'] == 'arima_ngarch':
    st.markdown('<div class="main-header">💹 Prediksi ARIMA-NGARCH dengan Model Volatilitas Mata Uang</div>', unsafe_allow_html=True)
    st.markdown("""
        Unggah data time series mata uang Anda untuk melakukan analisis dan prediksi menggunakan model ARIMA dan NGARCH.  
        Dashboard ini memudahkan visualisasi hasil prediksi dan evaluasi model volatilitas.
    """)
    # Tambahkan elemen UI khusus ARIMA-NGARCH Anda di sini, contoh:
    st.subheader("Unggah Data Mata Uang")
    uploaded_file = st.file_uploader("Pilih file CSV data mata uang", type=["csv"])
    if uploaded_file is not None:
        st.success("File berhasil diunggah!")
        # Anda kemudian akan memuat dan memproses data di sini
        # Untuk demonstrasi, hanya menampilkan pesan
        st.write("Data akan diproses untuk analisis ARIMA-NGARCH.")

    st.subheader("Hasil Prediksi dan Visualisasi")
    st.info("Area ini akan menampilkan grafik prediksi, volatilitas, dan metrik evaluasi model.")
    # Placeholder contoh untuk grafik/tabel
    # st.line_chart(beberapa_data_untuk_prediksi)
    # st.write(beberapa_dataframe_untuk_volatilitas)

# Anda juga dapat menambahkan blok elif untuk item menu lainnya
# elif st.session_state['current_page'] == 'input_data':
#     st.header("Input Data")
#     st.write("Di sinilah Anda akan mengunggah data Anda.")
#     # ... dan seterusnya untuk halaman lain

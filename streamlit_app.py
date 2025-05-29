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

import streamlit as st
import pandas as pd

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
        /* Styling untuk tombol yang saat ini dipilih (dari session state) */
        /* Perhatian: Ini adalah pendekatan custom dan mungkin tidak bekerja sempurna
           dengan semua versi Streamlit atau browser tanpa JavaScript tambahan.
           Streamlit umumnya mengelola state tombolnya sendiri.
           Untuk styling yang lebih andal, pertimbangkan untuk menggunakan Streamlit components
           atau memanipulasi DOM dengan JavaScript injeksi jika memungkinkan.
        */
        .stButton button[aria-selected="true"] { /* Custom attribute for active state */
            background-color: #d0d2d6 !important; /* Sedikit lebih gelap untuk status aktif */
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

# Logika untuk tombol sidebar agar tetap "aktif" secara visual
for item, key in menu_items.items():
    # Menambahkan atribut kustom untuk styling CSS
    # Perbaikan: Gunakan tanda kutip tunggal untuk nilai di dalam string ganda
    # dan gunakan st.sidebar.markdown dengan unsafe_allow_html=True untuk
    # menampilkan tombol dengan styling kustom, dan gunakan st.session_state
    # untuk menangani navigasi ketika tombol tersebut diklik.
    
    # Pendekatan yang lebih aman untuk styling tombol aktif di Streamlit adalah
    # dengan mengandalkan CSS selektor yang ada atau menggunakannya dengan hati-hati.
    # Karena st.button adalah komponen, memanipulasi HTML-nya secara langsung
    # seringkali tidak disarankan atau tidak efektif.

    # Namun, jika Anda ingin menjaga tampilan tombol yang "aktif" seperti di gambar,
    # kita bisa menggunakan st.columns dan st.button di dalamnya dengan sedikit trik CSS.
    
    # Alternatif sederhana tanpa HTML kustom yang rumit di st.button:
    # Anda bisa menggunakan warna latar belakang langsung di tombol Streamlit
    # jika itu cukup. Namun, untuk meniru tampilan gambar, CSS lebih baik.
    
    # Untuk mengatasi SyntaxError, kita akan memperbaiki f-string
    # Namun, perlu diingat bahwa menginject HTML custom ke st.button seringkali
    # tidak bekerja seperti yang diharapkan karena Streamlit merender komponennya.
    # Solusi terbaik untuk styling yang kompleks adalah dengan CSS class yang benar
    # atau custom component.
    
    # Untuk tujuan perbaikan SyntaxError:
    # Kita akan tetap menggunakan st.button dan mencoba menyinkronkan tampilan
    # aktif melalui CSS yang lebih umum atau, jika benar-benar ingin HTML kustom,
    # gunakan st.markdown untuk tombol buatan tangan (tetapi kemudian kehilangan
    # fungsionalitas st.button native).

    # Mari kita coba pendekatan yang lebih "Streamlit-friendly" untuk tombol aktif:
    # Kita akan tetap menggunakan st.button dan biarkan CSS menangani styling
    # berdasarkan apakah tombolnya 'aktif' atau tidak.
    
    # Hapus baris button_html = f'<button style=...>' karena itu menyebabkan error
    # dan biasanya tidak direkomendasikan untuk menimpa st.button secara langsung.
    
    # Kita hanya perlu tombol Streamlit biasa.
    if st.sidebar.button(item, key=key):
        st.session_state['current_page'] = key
    
    # Untuk membuat tombol aktif secara visual (seperti yang Anda inginkan di CSS):
    # Ini memerlukan sedikit trik karena Streamlit tidak menyediakan class 'active' secara langsung
    # Kita bisa menambahkan JavaScript melalui st.markdown untuk menambahkan class
    # atau mengubah style, tapi itu jauh lebih kompleks.
    # Untuk saat ini, kita akan fokus pada fungsionalitas dan mencegah SyntaxError.
    
    # Solusi paling bersih untuk styling tombol aktif secara dinamis
    # adalah dengan mengatur style berdasarkan kondisi dalam loop:
    if st.session_state['current_page'] == key:
        # Ini akan menimpa gaya default tombol jika kita mencoba menampilkannya lagi,
        # yang tidak ideal. Streamlit sudah merender tombol di st.button.
        # Jadi, cara terbaik adalah dengan CSS selektor yang menargetkan
        # tombol yang sedang aktif.

        # Saya akan menghapus baris `button_html` yang menyebabkan masalah
        # dan mempertahankan `st.sidebar.button`.
        # Untuk styling aktif, saya akan mengandalkan selektor CSS yang lebih umum
        # jika itu memungkinkan, atau menunjukkan cara lain.

        # Mari kita coba sedikit modifikasi CSS untuk menangani tombol yang aktif.
        # Streamlit memberikan ID atau class yang bisa kita target.
        # Biasanya tombol aktif akan mendapatkan fokus atau state tertentu.

        # Hapus baris yang menyebabkan SyntaxError.
        # button_html = f'<button style="width:100%; border-radius:0.5rem; border:1px solid #ddd; background-color:{"#d0d2d6" if st.session_state["current_page"] == key else "#f0f2f6"}; color:#333; padding:0.75rem 1rem; font-size:1rem; text-align:left; margin-bottom:0.2rem;" {"aria-selected='true'" if st.session_state["current_page"] == key else ""}>{item}</button>'
        # Ini tidak akan digunakan, jadi hapus saja.

        pass # `pass` karena kita tidak lagi membuat HTML tombol secara manual di sini.

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
        <li><b>PEMODELAN ARIMA:</b> Langkah-langkah untuk membentuk model ARIMA.</li>
        <li><b>PEMODELAN ANFIS ABC:</b> Langkah-langkah untuk membentuk model ANFIS dengan optimasi ABC.</li>
        <li><b>PEMODELAN ARIMA-ANFIS ABC:</b> Integrasi model ARIMA dan ANFIS ABC.</li>
        <li><b>PREDIKSI:</b> Menampilkan hasil prediksi dari model yang telah dibuat.</li>
        <li><b>ARIMA-NGARCH:</b> Bagian khusus untuk prediksi menggunakan model ARIMA-NGARCH dan volatilitas mata uang.</li>
    </ul>
    """, unsafe_allow_html=True)

elif st.session_state['current_page'] == 'arima_ngarch':
    st.markdown('<div class="main-header">💹 Prediksi ARIMA-NGARCH dengan Model Volatilitas Mata Uang</div>', unsafe_allow_html=True)
    st.markdown("""
        Unggah data time series mata uang Anda untuk melakukan analisis dan prediksi menggunakan model ARIMA dan NGARCH.
        Dashboard ini memudahkan visualisasi hasil prediksi dan evaluasi model volatilitas.
    """)
    st.subheader("Unggah Data Mata Uang")
    uploaded_file = st.file_uploader("Pilih file CSV data mata uang", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File berhasil diunggah dan dibaca!")
            st.write("5 baris pertama dari data Anda:")
            st.dataframe(df.head())

            st.session_state['df_currency'] = df

            st.write("Data siap untuk analisis ARIMA-NGARCH.")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.warning("Pastikan file yang diunggah adalah file CSV yang valid dan diformat dengan benar.")

    st.subheader("Hasil Prediksi dan Visualisasi")
    st.info("Area ini akan menampilkan grafik prediksi, volatilitas, dan metrik evaluasi model.")
    if 'df_currency' in st.session_state and not st.session_state['df_currency'].empty:
        st.write("Contoh tampilan data yang diunggah:")
        st.line_chart(st.session_state['df_currency'].iloc[:, 0])
    else:
        st.info("Unggah data untuk melihat contoh visualisasi.")

elif st.session_state['current_page'] == 'input_data':
    st.markdown('<div class="main-header">Input Data</div>', unsafe_allow_html=True)
    st.write("Di sinilah Anda dapat mengunggah berbagai jenis data time series untuk aplikasi Anda.")
    uploaded_file_general = st.file_uploader("Pilih file data (CSV, Excel, dll.)", type=["csv", "xlsx", "txt"])
    if uploaded_file_general is not None:
        try:
            if uploaded_file_general.name.endswith('.csv'):
                df_general = pd.read_csv(uploaded_file_general)
            elif uploaded_file_general.name.endswith(('.xls', '.xlsx')):
                df_general = pd.read_excel(uploaded_file_general)
            else:
                st.warning("Format file tidak didukung. Harap unggah file CSV atau Excel.")
                df_general = pd.DataFrame()

            if not df_general.empty:
                st.success("File berhasil diunggah dan dibaca!")
                st.write("5 baris pertama dari data Anda:")
                st.dataframe(df_general.head())
                st.session_state['df_general'] = df_general
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.warning("Pastikan file yang diunggah adalah file yang valid dan diformat dengan benar.")

elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">Data Preprocessing</div>', unsafe_allow_html=True)
    st.write("Lakukan langkah-langkah pembersihan dan persiapan data di bagian ini, seperti penanganan nilai hilang, normalisasi, dll.")
    if 'df_general' in st.session_state and not st.session_state['df_general'].empty:
        st.write("Data yang tersedia untuk preprocessing:")
        st.dataframe(st.session_state['df_general'].head())
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

elif st.session_state['current_page'] == 'pemodelan_arima_anfis_abc':
    st.markdown('<div class="main-header">Pemodelan ARIMA-ANFIS ABC</div>', unsafe_allow_html=True)
    st.write("Integrasi dan pelatihan model gabungan ARIMA dan ANFIS ABC.")

elif st.session_state['current_page'] == 'prediksi':
    st.markdown('<div class="main-header">Prediksi</div>', unsafe_allow_html=True)
    st.write("Lihat dan evaluasi hasil prediksi dari model yang telah Anda latih.")

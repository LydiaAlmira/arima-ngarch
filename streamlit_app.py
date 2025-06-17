import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from arch import arch_model

# --- Page Configuration ---
st.set_page_config(
    page_title="Prediksi Nilai Tukar & Volatilitas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #2F80ED;
        text-align: center;
        margin-bottom: 30px;
        padding: 15px;
        background-color: #e0f2f7;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8em;
        font-weight: bold;
        color: #4CAF50;
        margin-top: 25px;
        margin-bottom: 15px;
        border-bottom: 2px solid #4CAF50;
        padding-bottom: 5px;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .stDownloadButton>button {
        background-color: #007bff;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stDownloadButton>button:hover {
        background-color: #0056b3;
    }
    .st-emotion-cache-1r6dmc7 { /* For the overall container of main content */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    .st-emotion-cache-vk33gh { /* For sidebar text */
        font-size: 1.1em;
        color: #333;
    }
    .st-emotion-cache-1jm6as2 { /* For sidebar header */
        font-size: 1.5em;
        font-weight: bold;
        color: #1a1a1a;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Session State Initialization ---
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'home'
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'processed_data' not in st.session_state:
    st.session_state['processed_data'] = None
if 'selected_currency' not in st.session_state:
    st.session_state['selected_currency'] = None
if 'train_data_prices' not in st.session_state:
    st.session_state['train_data_prices'] = pd.Series()
if 'test_data_prices' not in st.session_state:
    st.session_state['test_data_prices'] = pd.Series()
if 'model_arima_fit' not in st.session_state:
    st.session_state['model_arima_fit'] = None
if 'arima_residuals' not in st.session_state:
    st.session_state['arima_residuals'] = pd.Series()
if 'arima_residual_has_arch_effect' not in st.session_state:
    st.session_state['arima_residual_has_arch_effect'] = None
if 'model_ngarch_fit' not in st.session_state:
    st.session_state['model_ngarch_fit'] = None
if 'future_predicted_prices_series' not in st.session_state:
    st.session_state['future_predicted_prices_series'] = pd.Series()
if 'future_predicted_volatility_series' not in st.session_state:
    st.session_state['future_predicted_volatility_series'] = pd.Series()
if 'predicted_prices_series' not in st.session_state: # Combined fitted + forecast for general plotting/evaluation
    st.session_state['predicted_prices_series'] = pd.Series()

# --- Sidebar Navigation ---
st.sidebar.title("Navigasi Aplikasi 🚀")
pages = {
    "Beranda": "home",
    "1. Unggah Data": "data_upload",
    "2. Pra-pemrosesan Data": "data_preprocessing",
    "3. Pembagian Data (Train/Test)": "data_splitting",
    "4. Model & Prediksi ARIMA": "arima_modeling_prediction",
    "5. Model & Prediksi NGARCH": "ngarch_modeling_prediction",
    "6. Evaluasi Model": "evaluation"
}

for page_name, page_id in pages.items():
    if st.sidebar.button(page_name, key=f"nav_{page_id}"):
        st.session_state['current_page'] = page_id

# --- Main Content Area ---
if st.session_state['current_page'] == 'home':
    st.markdown('<div class="main-header">SELAMAT DATANG DI APLIKASI PREDIKSI NILAI TUKAR MATA UANG! 🌐📊</div>', unsafe_allow_html=True)
    st.write("""
        Aplikasi ini dirancang untuk memprediksi nilai tukar mata uang menggunakan model **ARIMA (Autoregressive Integrated Moving Average)** untuk memodelkan bagian mean (harga) dan **NGARCH (Nonlinear Generalized Autoregressive Conditional Heteroskedasticity)** untuk memodelkan bagian varians (volatilitas).

        Ikuti langkah-langkah di sidebar untuk memulai prediksi Anda:
        1.  **Unggah Data:** Unggah dataset nilai tukar mata uang Anda.
        2.  **Pra-pemrosesan Data:** Bersihkan dan siapkan data untuk analisis.
        3.  **Pembagian Data (Train/Test):** Pisahkan data Anda menjadi set pelatihan dan pengujian.
        4.  **Model & Prediksi ARIMA:** Latih model ARIMA untuk memprediksi harga.
        5.  **Model & Prediksi NGARCH:** Latih model NGARCH untuk memprediksi volatilitas berdasarkan residual ARIMA.
        6.  **Evaluasi Model:** Lihat metrik evaluasi untuk kinerja model.

        Selamat menganalisis!
    """)

elif st.session_state['current_page'] == 'data_upload':
    st.markdown('<div class="main-header">UNGGAH DATA 📥📈</div>', unsafe_allow_html=True)
    st.write("Silakan unggah file CSV atau Excel yang berisi data nilai tukar mata uang Anda.")

    uploaded_file = st.file_uploader("Pilih file CSV atau Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else: # .xlsx
                df = pd.read_excel(uploaded_file)

            st.session_state['uploaded_data'] = df
            st.success("File berhasil diunggah! 🎉")
            st.write("Pratinjau Data:")
            st.dataframe(df.head())
            st.info("Anda dapat melanjutkan ke langkah 'Pra-pemrosesan Data' di sidebar.")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e} ❌ Pastikan format file benar.")

elif st.session_state['current_page'] == 'data_preprocessing':
    st.markdown('<div class="main-header">PRA-PEMROSESAN DATA 🧹📊</div>', unsafe_allow_html=True)
    st.write("Lakukan pra-pemrosesan data seperti memilih kolom, mengubah ke format datetime, dan menangani nilai yang hilang.")

    if st.session_state['uploaded_data'] is not None:
        df = st.session_state['uploaded_data'].copy()

        st.subheader("A. Pilih Kolom Tanggal dan Harga 🗓️💲")
        date_column = st.selectbox("Pilih kolom yang berisi Tanggal:", df.columns, key="date_col_select")
        price_column = st.selectbox("Pilih kolom yang berisi Harga Nilai Tukar:", df.columns, key="price_col_select")

        if date_column and price_column:
            st.subheader("B. Konversi Kolom Tanggal ke Format Datetime 🔄")
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                df.set_index(date_column, inplace=True)
                df = df[[price_column]]
                df.columns = ['Price'] # Rename the price column to 'Price' for consistency
                st.success(f"Kolom '{date_column}' berhasil dikonversi ke datetime dan dijadikan indeks. Kolom '{price_column}' dipilih sebagai 'Price'. ✅")
            except Exception as e:
                st.error(f"Gagal mengonversi kolom tanggal atau mengatur indeks: {e} ❌ Harap pastikan format tanggal benar.")
                st.stop()

            st.subheader("C. Tangani Nilai yang Hilang (Missing Values) 🗑️")
            missing_values_count = df.isnull().sum()
            st.write("Jumlah nilai yang hilang per kolom:")
            st.dataframe(missing_values_count.to_frame(name='Missing Values Count'))

            if missing_values_count.sum() > 0:
                filling_method = st.radio(
                    "Pilih metode untuk menangani nilai yang hilang:",
                    ("Interpolasi (Linear)", "Hapus Baris"),
                    key="missing_value_method"
                )
                if filling_method == "Interpolasi (Linear)":
                    df.interpolate(method='linear', inplace=True)
                    st.success("Nilai yang hilang telah diinterpolasi secara linear. ✅")
                else:
                    df.dropna(inplace=True)
                    st.success("Baris dengan nilai yang hilang telah dihapus. ✅")
            else:
                st.info("Tidak ada nilai yang hilang dalam dataset. 👍")

            # Remove duplicates based on index (date)
            if df.index.duplicated().any():
                st.warning("Tanggal duplikat ditemukan. Menghapus duplikat dan mempertahankan entri terakhir. ⚠️")
                df = df[~df.index.duplicated(keep='last')]
                st.success("Tanggal duplikat berhasil dihapus. ✅")

            # Sort by index (date)
            df.sort_index(inplace=True)
            st.success("Data berhasil diurutkan berdasarkan tanggal. ✅")

            st.subheader("D. Pilih Mata Uang (Kolom 'Price') 💰")
            st.info("Untuk memudahkan identifikasi, Anda bisa memberi label mata uang untuk kolom 'Price'.")
            currency_options = ["IDR", "SGD", "MYR", "THB", "PHP"]
            selected_currency = st.selectbox("Pilih mata uang untuk kolom 'Price':", currency_options, key="currency_selector")
            st.session_state['selected_currency'] = selected_currency
            st.write(f"Mata uang yang dipilih: **{selected_currency}**")

            # Store processed data
            st.session_state['processed_data'] = df
            st.write("Pratinjau Data Setelah Pra-pemrosesan:")
            st.dataframe(df.head())
            st.info("Pra-pemrosesan data selesai. Anda dapat melanjutkan ke langkah 'Pembagian Data (Train/Test)'.")
    else:
        st.warning("Belum ada data yang diunggah. Harap kembali ke 'Unggah Data' terlebih dahulu. ⚠️")

elif st.session_state['current_page'] == 'data_splitting':
    st.markdown('<div class="main-header">PEMBAGIAN DATA (TRAIN/TEST) ✂️📊</div>', unsafe_allow_html=True)
    st.write("Bagi data yang telah diproses menjadi set pelatihan dan pengujian untuk evaluasi model.")

    if st.session_state['processed_data'] is not None and not st.session_state['processed_data'].empty:
        df_prices = st.session_state['processed_data']['Price']

        st.subheader("A. Visualisasi Data Harga Asli 📈")
        fig_raw_prices = go.Figure()
        fig_raw_prices.add_trace(go.Scatter(x=df_prices.index, y=df_prices.values, mode='lines', name='Harga Asli'))
        fig_raw_prices.update_layout(title_text=f'Data Harga {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_raw_prices)

        st.subheader("B. Tetapkan Ukuran Pembagian Data 📏")
        train_size_ratio = st.slider(
            "Pilih rasio data pelatihan (%):",
            min_value=60,
            max_value=90,
            value=80,
            step=5,
            key="train_size_slider"
        )
        train_size = int(len(df_prices) * (train_size_ratio / 100))

        train_data_prices = df_prices.iloc[:train_size]
        test_data_prices = df_prices.iloc[train_size:]

        st.session_state['train_data_prices'] = train_data_prices
        st.session_state['test_data_prices'] = test_data_prices

        st.write(f"Ukuran Data Total: **{len(df_prices)}**")
        st.write(f"Ukuran Data Pelatihan: **{len(train_data_prices)}** ({train_size_ratio}%)")
        st.write(f"Ukuran Data Pengujian: **{len(test_data_prices)}** ({100 - train_size_ratio}%)")

        st.subheader("C. Visualisasi Pembagian Data 📊")
        fig_split = go.Figure()
        fig_split.add_trace(go.Scatter(x=train_data_prices.index, y=train_data_prices.values, mode='lines', name='Data Pelatihan'))
        fig_split.add_trace(go.Scatter(x=test_data_prices.index, y=test_data_prices.values, mode='lines', name='Data Pengujian'))
        fig_split.update_layout(title_text=f'Pembagian Data Harga {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_split)

        st.success("Data berhasil dibagi menjadi set pelatihan dan pengujian. ✅")
        st.info("Anda dapat melanjutkan ke langkah 'Model & Prediksi ARIMA'.")

    else:
        st.warning("Data yang diproses belum tersedia. Harap selesaikan langkah 'Unggah Data' dan 'Pra-pemrosesan Data' terlebih dahulu. ⚠️")

elif st.session_state['current_page'] == 'arima_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI ARIMA 📈🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model ARIMA pada data harga awal {st.session_state.get('selected_currency', '')} untuk memodelkan mean (prediksi nilai tukar), lalu lakukan prediksi. 📊")

    # Pastikan variabel-variabel ini tersedia di session_state dari halaman sebelumnya
    train_data_prices = st.session_state.get('train_data_prices', pd.Series())
    test_data_prices = st.session_state.get('test_data_prices', pd.Series())

    if not train_data_prices.empty and not test_data_prices.empty:
        # --- MODEL ARIMA SECTION ---
        st.markdown("<h3 class='section-header'>A. Pemodelan ARIMA (Mean Equation) ⚙️</h3>", unsafe_allow_html=True)
        st.write(f"Data pelatihan harga awal untuk pemodelan ARIMA ({st.session_state.get('selected_currency', '')}):")
        st.dataframe(train_data_prices.head())

        st.subheader("A.1. Pilih Ordo ARIMA (p, d, q) 🔢")
        st.info("Pilih kombinasi ordo ARIMA yang telah ditentukan. Ordo differencing (d) akan digunakan untuk membuat data stasioner jika diperlukan.")

        arima_orders_options = {
            "ARIMA (1,1,1)": (1, 1, 1),
            "ARIMA (0,1,1)": (0, 1, 1),
            "ARIMA (1,1,0)": (1, 1, 0),
            "ARIMA (2,1,0)": (2, 1, 0),
            "ARIMA (0,1,2)": (0, 1, 2),
            "ARIMA (2,1,1)": (2, 1, 1),
            "ARIMA (1,1,2)": (1, 1, 2),
            "ARIMA (2,1,2)": (2, 1, 2)
        }
        
        selected_arima_label = st.selectbox(
            "Pilih salah satu model ARIMA:",
            list(arima_orders_options.keys()),
            key="arima_model_selector"
        )
        
        p, d, q = arima_orders_options[selected_arima_label]
        
        st.write(f"Ordo ARIMA yang dipilih: **p={p}, d={d}, q={q}**")

        if st.button("A.2. Latih Model ARIMA ▶️", key="train_arima_button"):
            try:
                with st.spinner("Melatih model ARIMA... ⏳"):
                    model_arima = ARIMA(train_data_prices, order=(p, d, q))
                    model_arima_fit = model_arima.fit()

                    st.session_state['model_arima_fit'] = model_arima_fit
                    st.session_state['arima_residuals'] = pd.Series(model_arima_fit.resid, index=train_data_prices.index)
                    st.success("Model ARIMA berhasil dilatih! 🎉")
                    
                    st.subheader("A.3. Ringkasan Model ARIMA (Koefisien dan Statistik) 📝")
                    st.text(model_arima_fit.summary().as_text())

                    st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                    st.info("P-value untuk setiap koefisien menunjukkan signifikansi statistik. Koefisien dianggap signifikan jika P-value < 0.05 (pada tingkat kepercayaan 95%).")
                    
                    model_summary = model_arima_fit.summary()
                    # Ensure tables exist
                    if hasattr(model_summary, 'tables') and len(model_summary.tables) > 1:
                        params_table = model_summary.tables[1]
                        st.dataframe(params_table)
                    else:
                        st.info("Tidak dapat menampilkan tabel koefisien secara rinci. Silakan lihat ringkasan model di atas.")

                    # --- PREDIKSI HARGA DENGAN ARIMA ---
                    st.markdown("<h3 class='section-header'>B. Prediksi Harga dengan ARIMA 📈🔮</h3>", unsafe_allow_html=True)
                    st.info("Prediksi harga akan dilakukan pada data pengujian. Karena model ARIMA dilatih pada data harga awal dengan differencing (d>0), prediksi secara otomatis akan direkonstruksi ke skala harga asli.")

                    # Fitted values on training data
                    fitted_values_arima = model_arima_fit.predict(start=0, end=len(train_data_prices)-1, typ='levels')
                    fitted_values_arima.name = 'Fitted Values'

                    # Forecast on test data
                    forecast_steps = len(test_data_prices)
                    forecast_arima = model_arima_fit.forecast(steps=forecast_steps, typ='levels')
                    forecast_arima.index = test_data_prices.index # Assign test data index to forecast

                    st.session_state['last_forecast_price_arima'] = forecast_arima.iloc[-1]
                    st.session_state['future_predicted_prices_series'] = forecast_arima # This is the out-of-sample forecast
                    
                    # Store predicted_prices_series (gabungan) for plotting/eval
                    predicted_prices_combined = pd.concat([fitted_values_arima, forecast_arima])
                    st.session_state['predicted_prices_series'] = predicted_prices_combined
                    
                    st.success("Prediksi harga dengan ARIMA selesai! ✅")
                    st.write(f"Prediksi harga terakhir: {st.session_state['last_forecast_price_arima']:.4f}")

                    st.subheader("B.1. Visualisasi Prediksi Harga ARIMA 📊")
                    fig_arima_pred = go.Figure()
                    fig_arima_pred.add_trace(go.Scatter(x=train_data_prices.index, y=train_data_prices.values, mode='lines', name='Data Pelatihan (Harga Asli)', line=dict(color='#3f72af')))
                    fig_arima_pred.add_trace(go.Scatter(x=test_data_prices.index, y=test_data_prices.values, mode='lines', name='Data Pengujian (Harga Asli)', line=dict(color='#ff7f0e')))
                    fig_arima_pred.add_trace(go.Scatter(x=fitted_values_arima.index, y=fitted_values_arima.values, mode='lines', name='ARIMA Fitted (Train)', line=dict(color='green', dash='dot')))
                    fig_arima_pred.add_trace(go.Scatter(x=forecast_arima.index, y=forecast_arima.values, mode='lines', name='ARIMA Forecast (Test)', line=dict(color='purple', dash='dash')))
                    fig_arima_pred.update_layout(title_text=f'Prediksi Harga {st.session_state.get("selected_currency", "")} dengan ARIMA', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_arima_pred)

                    # --- Penambahan untuk Menyimpan Prediksi ARIMA ---
                    st.subheader("B.2. Simpan Hasil Prediksi Harga ARIMA 💾")
                    if not st.session_state['future_predicted_prices_series'].empty:
                        # Membuat DataFrame untuk disimpan
                        df_arima_forecast = pd.DataFrame(st.session_state['future_predicted_prices_series'])
                        df_arima_forecast.columns = [f'Predicted_Price_{st.session_state.get("selected_currency", "Value")}']
                        
                        # Menyiapkan file untuk di-download
                        csv_file = df_arima_forecast.to_csv(index=True).encode('utf-8')
                        st.download_button(
                            label="Unduh Prediksi Harga ARIMA (.csv) ⬇️",
                            data=csv_file,
                            file_name=f'Prediksi_Harga_ARIMA_{st.session_state.get("selected_currency", "Value")}.csv',
                            mime='text/csv',
                            key="download_arima_forecast_button"
                        )
                        st.info("Tombol unduh tersedia. Klik untuk menyimpan hasil prediksi harga ARIMA.")
                    else:
                        st.warning("Tidak ada prediksi harga ARIMA yang tersedia untuk disimpan. Harap latih dan prediksi model terlebih dahulu. ⚠️")

                    # --- UJI ASUMSI PADA RESIDUAL ARIMA ---
                    st.markdown("<h3 class='section-header'>C. Uji Asumsi pada Residual ARIMA 📊🧪</h3>", unsafe_allow_html=True)
                    st.info("Setelah memodelkan mean (harga) dengan ARIMA, kita perlu memeriksa residualnya untuk volatilitas. Jika residual menunjukkan efek ARCH (heteroskedastisitas), maka model GARCH/NGARCH cocok untuk memodelkan varians.")

                    # Visualisasi Residual ARIMA
                    st.subheader("C.1. Plot Residual ARIMA 📈")
                    fig_resid = go.Figure()
                    fig_resid.add_trace(go.Scatter(x=st.session_state['arima_residuals'].index, y=st.session_state['arima_residuals'].values, mode='lines', name='Residual ARIMA', line=dict(color='#8856a7')))
                    fig_resid.update_layout(title_text=f'Residual Model ARIMA {st.session_state.get("selected_currency", "")}', xaxis_rangeslider_visible=True)
                    st.plotly_chart(fig_resid)

                    # Plot ACF dan PACF dari Residual Kuadrat
                    st.subheader("C.2. Plot ACF dan PACF Residual Kuadrat (untuk ARCH Effect) 📈📉")
                    st.info("Jika ada pola yang signifikan pada plot ACF/PACF dari residual kuadrat, itu mengindikasikan adanya efek ARCH/GARCH (volatilitas clustering).")
                    
                    if not st.session_state['arima_residuals'].empty:
                        # Hapus baris dengan NaN atau inf jika ada
                        residuals_clean = st.session_state['arima_residuals'].dropna()
                        residuals_squared = residuals_clean**2
                        lags = min(20, len(residuals_squared) // 2 - 1) # Ensure lags are reasonable

                        if not residuals_squared.empty and lags > 0:
                            # Use matplotlib directly for ACF/PACF plots as Streamlit can display them
                            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
                            plot_acf(residuals_squared, lags=lags, ax=ax[0], alpha=0.05)
                            ax[0].set_title(f'ACF Residual Kuadrat ARIMA {st.session_state.get("selected_currency", "")}')
                            plot_pacf(residuals_squared, lags=lags, ax=ax[1], alpha=0.05)
                            ax[1].set_title(f'PACF Residual Kuadrat ARIMA {st.session_state.get("selected_currency", "")}')
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig) # Close the figure to prevent duplicate plots
                        else:
                            st.warning("Residual kuadrat kosong atau tidak valid untuk diplot ACF/PACF, atau lags tidak cukup.")
                    else:
                        st.warning("Residual ARIMA belum tersedia untuk plotting. Latih model ARIMA terlebih dahulu.")


                    # Uji Ljung-Box pada Residual (White Noise Test)
                    st.subheader("C.3. Uji Ljung-Box pada Residual ARIMA (Autokorelasi) 🎲")
                    st.info("Menguji apakah residual adalah white noise (tidak ada autokorelasi). P-value > 0.05 menunjukkan residual adalah white noise.")
                    if not st.session_state['arima_residuals'].empty:
                        # Ljung-Box pada residual asli
                        if len(st.session_state['arima_residuals']) > 10: # Need enough data for lags=10
                            lb_test = sm.stats.diagnostic.acorr_ljungbox(st.session_state['arima_residuals'], lags=[10], return_df=True)
                            st.write("Hasil Uji Ljung-Box pada Residual:")
                            st.dataframe(lb_test)
                            if lb_test.iloc[0]['lb_pvalue'] > 0.05:
                                st.success("Residual ARIMA tidak menunjukkan autokorelasi yang signifikan (white noise). ✅")
                            else:
                                st.warning("Residual ARIMA masih menunjukkan autokorelasi yang signifikan. ⚠️ Mungkin ordo ARIMA perlu disesuaikan.")
                        else:
                            st.warning("Tidak cukup data residual untuk uji Ljung-Box dengan lags 10.")
                    else:
                        st.warning("Residual ARIMA belum tersedia untuk uji Ljung-Box.")

                    # Uji ARCH (Heteroskedastisitas)
                    st.subheader("C.4. Uji ARCH pada Residual ARIMA (Heteroskedastisitas) 🌪️")
                    st.info("Menguji adanya efek ARCH (volatilitas bervariasi seiring waktu). P-value < 0.05 menunjukkan ada efek ARCH, sehingga model GARCH/NGARCH cocok.")
                    if not st.session_state['arima_residuals'].empty:
                        # Uji ARCH pada residual
                        if len(st.session_state['arima_residuals']) > 1:
                            try:
                                arch_test_result = het_arch(st.session_state['arima_residuals'], nlags=10) # Gunakan nlags default 10 atau sesuaikan
                                st.write("Hasil Uji ARCH (Lagrange Multiplier Test):")
                                st.write(f"**Statistik Uji:** {arch_test_result[0]:.4f}")
                                st.write(f"**P-value:** {arch_test_result[1]:.4f}")
                                st.write(f"**F-Statistik:** {arch_test_result[2]:.4f}")
                                st.write(f"**F-Pvalue:** {arch_test_result[3]:.4f}")

                                if arch_test_result[1] < 0.05:
                                    st.success("Ada efek ARCH/Heteroskedastisitas yang signifikan pada residual ARIMA (P-value < 0.05). 🎉 Ini mengindikasikan bahwa model GARCH/NGARCH cocok untuk memodelkan volatilitas.")
                                    st.session_state['arima_residual_has_arch_effect'] = True
                                else:
                                    st.info("Tidak ada efek ARCH/Heteroskedastisitas yang signifikan (P-value >= 0.05). ℹ️ Model GARCH/NGARCH mungkin tidak diperlukan untuk memodelkan volatilitas residual ini.")
                                    st.session_state['arima_residual_has_arch_effect'] = False
                            except Exception as e:
                                st.error(f"Gagal menjalankan Uji ARCH: {e}. Pastikan residual tidak kosong atau berisi nilai tidak valid.")
                                st.session_state['arima_residual_has_arch_effect'] = None
                        else:
                            st.warning("Tidak cukup data residual untuk melakukan Uji ARCH.")
                            st.session_state['arima_residual_has_arch_effect'] = None
                    else:
                        st.warning("Residual ARIMA belum tersedia untuk uji ARCH.")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melatih model ARIMA atau melakukan prediksi: {e} ❌ Harap pastikan data dan ordo ARIMA Anda sesuai.")
    else:
        st.warning("Data pelatihan atau pengujian belum tersedia. Silakan selesaikan langkah 'Data Splitting' terlebih dahulu. ⚠️")

# --- Bagian untuk NGARCH ---
elif st.session_state['current_page'] == 'ngarch_modeling_prediction':
    st.markdown('<div class="main-header">MODEL & PREDIKSI NGARCH 🌪️🔮</div>', unsafe_allow_html=True)
    st.write(f"Latih model NGARCH pada residual kuadrat dari model ARIMA untuk memodelkan volatilitas {st.session_state.get('selected_currency', '')}, lalu lakukan prediksi volatilitas. 📊")

    arima_residuals = st.session_state.get('arima_residuals', pd.Series())
    # Pastikan residual ada dan punya efek ARCH
    if arima_residuals.empty or not st.session_state.get('arima_residual_has_arch_effect', False):
        st.warning("Model NGARCH hanya dapat dijalankan jika residual ARIMA tersedia dan menunjukkan efek ARCH yang signifikan. Harap selesaikan bagian 'Model & Prediksi ARIMA' dan pastikan uji ARCH berhasil. ⚠️")
        st.stop()

    st.write(f"Residual ARIMA yang akan digunakan untuk pemodelan NGARCH:")
    st.dataframe(arima_residuals.head())

    st.subheader("A.1. Pilih Ordo NGARCH (p, o, q) 🔢")
    st.info("Pilih kombinasi ordo NGARCH (p, o, q) yang sesuai untuk memodelkan volatilitas. 'p' adalah ordo ARCH, 'o' adalah ordo asimetri (leverage effect), dan 'q' adalah ordo GARCH.")

    ngarch_orders_options = {
        "NGARCH (1,1,1)": (1, 1, 1), # Default NGARCH
        "NGARCH (1,0,1)": (1, 0, 1), # Equivalent to GARCH(1,1) (no asymmetry)
        "NGARCH (2,1,1)": (2, 1, 1),
        "NGARCH (1,1,2)": (1, 1, 2)
    }

    selected_ngarch_label = st.selectbox(
        "Pilih salah satu model NGARCH:",
        list(ngarch_orders_options.keys()),
        key="ngarch_model_selector"
    )
    
    p_ngarch, o_ngarch, q_ngarch = ngarch_orders_options[selected_ngarch_label]
    
    st.write(f"Ordo NGARCH yang dipilih: **p={p_ngarch}, o={o_ngarch}, q={q_ngarch}**")

    if st.button("A.2. Latih Model NGARCH ▶️", key="train_ngarch_button"):
        try:
            with st.spinner("Melatih model NGARCH... ⏳"):
                # Use arch_model for NGARCH. It expects the residuals/returns directly.
                residuals_for_ngarch = arima_residuals.astype(float)

                # Initialize NGARCH model
                ngarch_model = arch_model(residuals_for_ngarch,
                                          mean='Zero', # Mean of residuals is zero after ARIMA
                                          vol='NGARCH',
                                          p=p_ngarch,
                                          o=o_ngarch,
                                          q=q_ngarch,
                                          dist='normal') # Bisa juga StudentsT atau SkewStudent
                
                ngarch_res = ngarch_model.fit(disp='off') # disp='off' untuk mengurangi output di Streamlit

                st.session_state['model_ngarch_fit'] = ngarch_res
                st.success("Model NGARCH berhasil dilatih! 🎉")

                st.subheader("A.3. Ringkasan Model NGARCH (Koefisien dan Statistik) 📝")
                st.text(ngarch_res.summary().as_text())

                st.subheader("A.4. Uji Signifikansi Koefisien (P-value) ✅❌")
                st.info("Sama seperti ARIMA, P-value < 0.05 menunjukkan koefisien NGARCH signifikan secara statistik.")
                if hasattr(ngarch_res.summary(), 'tables') and len(ngarch_res.summary().tables) > 1:
                    params_table_ngarch = ngarch_res.summary().tables[1]
                    st.dataframe(params_table_ngarch)
                else:
                    st.info("Tidak dapat menampilkan tabel koefisien secara rinci. Silakan lihat ringkasan model di atas.")

                # --- PREDIKSI VOLATILITAS DENGAN NGARCH ---
                st.markdown("<h3 class='section-header'>B. Prediksi Volatilitas dengan NGARCH 🌪️🔮</h3>", unsafe_allow_html=True)
                st.info("Prediksi volatilitas (standar deviasi kondisional) akan dilakukan berdasarkan model NGARCH yang telah dilatih.")

                test_data_prices = st.session_state.get('test_data_prices', pd.Series())
                forecast_horizon_vol = len(test_data_prices) # Prediksi volatilitas untuk durasi yang sama dengan data uji harga
                
                ngarch_forecast = ngarch_res.forecast(horizon=forecast_horizon_vol, method='simulation', simulations=1000)
                
                forecasted_conditional_variance = ngarch_forecast.variance.iloc[-1]
                forecasted_volatility = np.sqrt(forecasted_conditional_variance)
                
                last_train_date = arima_residuals.index[-1]
                # Try to infer frequency from residuals index, otherwise default to 'D'
                freq = arima_residuals.index.freq
                if freq is None:
                    # If frequency is not explicitly set, try to infer from data spacing
                    if len(arima_residuals) > 1:
                        time_diff = arima_residuals.index[1] - arima_residuals.index[0]
                        if time_diff == pd.Timedelta(days=1):
                            freq = 'D'
                        elif time_diff == pd.Timedelta(weeks=1):
                            freq = 'W'
                        elif time_diff == pd.Timedelta(months=1):
                            freq = 'M'
                        else:
                            freq = 'D' # Default to daily if no clear pattern or only one data point
                    else:
                        freq = 'D' # Default to daily for very small datasets
                        
                forecast_dates = pd.date_range(start=last_train_date + pd.Timedelta(freq='D'), periods=forecast_horizon_vol, freq=freq)
                
                # Adjust forecast dates if they go beyond the test data period
                # Make sure the forecast dates align with the test data period
                if len(test_data_prices) > 0:
                    forecast_dates = pd.date_range(start=test_data_prices.index[0], periods=forecast_horizon_vol, freq=freq)
                    if len(forecast_dates) > len(test_data_prices):
                        forecast_dates = forecast_dates[:len(test_data_prices)]
                    elif len(forecast_dates) < len(test_data_prices):
                        # This scenario might happen if the inferred frequency is too sparse
                        # For simplicity, if actual test data length is different, adjust forecast_horizon_vol
                        forecast_horizon_vol = len(test_data_prices)
                        forecast_dates = pd.date_range(start=test_data_prices.index[0], periods=forecast_horizon_vol, freq=freq)

                future_predicted_volatility_series = pd.Series(forecasted_volatility.values, index=forecast_dates)
                
                st.session_state['last_forecast_volatility_ngarch'] = future_predicted_volatility_series.iloc[-1]
                st.session_state['future_predicted_volatility_series'] = future_predicted_volatility_series # This is the out-of-sample forecast
                
                st.success("Prediksi volatilitas dengan NGARCH selesai! ✅")
                st.write(f"Prediksi volatilitas terakhir: {st.session_state['last_forecast_volatility_ngarch']:.4f}")

                st.subheader("B.1. Visualisasi Prediksi Volatilitas NGARCH 📊")
                fig_ngarch_pred = go.Figure()
                
                historical_volatility = np.sqrt(ngarch_res.conditional_variance)
                fig_ngarch_pred.add_trace(go.Scatter(x=historical_volatility.index, y=historical_volatility.values, mode='lines', name='Volatilitas Historis (Fitted)', line=dict(color='#3f72af')))
                
                fig_ngarch_pred.add_trace(go.Scatter(x=future_predicted_volatility_series.index, y=future_predicted_volatility_series.values, mode='lines', name='NGARCH Forecast (Volatilitas)', line=dict(color='orange', dash='dash')))
                
                fig_ngarch_pred.update_layout(title_text=f'Prediksi Volatilitas {st.session_state.get("selected_currency", "")} dengan NGARCH', xaxis_rangeslider_visible=True)
                st.plotly_chart(fig_ngarch_pred)

                # --- Penambahan untuk Menyimpan Prediksi NGARCH ---
                st.subheader("B.2. Simpan Hasil Prediksi Volatilitas NGARCH 💾")
                if not st.session_state['future_predicted_volatility_series'].empty:
                    # Membuat DataFrame untuk disimpan
                    df_ngarch_forecast = pd.DataFrame(st.session_state['future_predicted_volatility_series'])
                    df_ngarch_forecast.columns = [f'Predicted_Volatility_{st.session_state.get("selected_currency", "Value")}']
                    
                    # Menyiapkan file untuk di-download
                    csv_file = df_ngarch_forecast.to_csv(index=True).encode('utf-8')
                    st.download_button(
                        label="Unduh Prediksi Volatilitas NGARCH (.csv) ⬇️",
                        data=csv_file,
                        file_name=f'Prediksi_Volatilitas_NGARCH_{st.session_state.get("selected_currency", "Value")}.csv',
                        mime='text/csv',
                        key="download_ngarch_forecast_button"
                    )
                    st.info("Tombol unduh tersedia. Klik untuk menyimpan hasil prediksi volatilitas NGARCH.")
                else:
                    st.warning("Tidak ada prediksi volatilitas NGARCH yang tersedia untuk disimpan. Harap latih dan prediksi model terlebih dahulu. ⚠️")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat melatih model NGARCH atau melakukan prediksi: {e} ❌ Harap pastikan residual ARIMA dan ordo NGARCH Anda sesuai.")
    else:
        st.warning("Residual ARIMA belum tersedia atau tidak menunjukkan efek ARCH. Pastikan Anda telah menyelesaikan langkah 'Model & Prediksi ARIMA' dan uji ARCH menunjukkan adanya efek. ⚠️")

elif st.session_state['current_page'] == 'evaluation':
    st.markdown('<div class="main-header">EVALUASI MODEL 📊✅</div>', unsafe_allow_html=True)
    st.write("Evaluasi kinerja model ARIMA dan NGARCH menggunakan berbagai metrik.")

    # --- Evaluasi ARIMA (Harga) ---
    st.markdown("<h3 class='section-header'>A. Evaluasi Prediksi Harga ARIMA 📈</h3>", unsafe_allow_html=True)
    if not st.session_state['test_data_prices'].empty and not st.session_state['future_predicted_prices_series'].empty:
        actual_prices = st.session_state['test_data_prices']
        predicted_prices = st.session_state['future_predicted_prices_series']

        # Ensure both series have the same index for accurate comparison
        # This is crucial if forecast_dates for NGARCH were adjusted
        actual_prices_aligned = actual_prices.reindex(predicted_prices.index).dropna()
        predicted_prices_aligned = predicted_prices.reindex(actual_prices_aligned.index).dropna()

        if not actual_prices_aligned.empty and not predicted_prices_aligned.empty:
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            from math import sqrt

            rmse = sqrt(mean_squared_error(actual_prices_aligned, predicted_prices_aligned))
            mae = mean_absolute_error(actual_prices_aligned, predicted_prices_aligned)
            mape = np.mean(np.abs((actual_prices_aligned - predicted_prices_aligned) / actual_prices_aligned)) * 100

            st.write(f"**Metrik Evaluasi Prediksi Harga ({st.session_state.get('selected_currency', '')}):**")
            st.write(f"- RMSE (Root Mean Squared Error): {rmse:.4f}")
            st.write(f"- MAE (Mean Absolute Error): {mae:.4f}")
            st.write(f"- MAPE (Mean Absolute Percentage Error): {mape:.2f}%")

            st.subheader("A.1. Visualisasi Aktual vs. Prediksi Harga 📊")
            fig_eval_price = go.Figure()
            fig_eval_price.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.values, mode='lines', name='Harga Aktual (Test)', line=dict(color='blue')))
            fig_eval_price.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices.values, mode='lines', name='Harga Prediksi (ARIMA)', line=dict(color='red', dash='dash')))
            fig_eval_price.update_layout(title_text=f'Harga Aktual vs. Prediksi ({st.session_state.get("selected_currency", "")})', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig_eval_price)
        else:
            st.warning("Data aktual atau prediksi harga tidak selaras untuk evaluasi. Harap periksa langkah sebelumnya. ⚠️")
    else:
        st.warning("Data pengujian atau prediksi harga ARIMA belum tersedia. Harap selesaikan langkah 'Model & Prediksi ARIMA' terlebih dahulu. ⚠️")

    # --- Evaluasi NGARCH (Volatilitas) ---
    st.markdown("<h3 class='section-header'>B. Evaluasi Prediksi Volatilitas NGARCH 🌪️</h3>", unsafe_allow_html=True)
    st.info("Evaluasi volatilitas lebih kompleks karena tidak ada 'aktual' volatilitas yang terobservasi secara langsung. Kita bisa membandingkannya dengan volatilitas realisasi atau secara kualitatif.")
    
    if st.session_state['model_ngarch_fit'] is not None and not st.session_state['future_predicted_volatility_series'].empty:
        # Visualisasi volatilitas historis vs prediksi
        st.subheader("B.1. Visualisasi Volatilitas Historis (Fitted) vs. Prediksi NGARCH 📊")
        fig_eval_volatility = go.Figure()
        
        historical_volatility = np.sqrt(st.session_state['model_ngarch_fit'].conditional_variance)
        fig_eval_volatility.add_trace(go.Scatter(x=historical_volatility.index, y=historical_volatility.values, mode='lines', name='Volatilitas Historis (Fitted)', line=dict(color='#3f72af')))
        fig_eval_volatility.add_trace(go.Scatter(x=st.session_state['future_predicted_volatility_series'].index, y=st.session_state['future_predicted_volatility_series'].values, mode='lines', name='Volatilitas Prediksi (NGARCH)', line=dict(color='orange', dash='dash')))
        fig_eval_volatility.update_layout(title_text=f'Volatilitas {st.session_state.get("selected_currency", "")} Historis vs. Prediksi', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig_eval_volatility)

        st.success("Visualisasi evaluasi volatilitas NGARCH tersedia. Untuk evaluasi kuantitatif lebih lanjut, diperlukan proxy volatilitas realisasi (misalnya, menggunakan data intraday).")
    else:
        st.warning("Model NGARCH belum dilatih atau prediksi volatilitas belum tersedia. Harap selesaikan langkah 'Model & Prediksi NGARCH' terlebih dahulu. ⚠️")

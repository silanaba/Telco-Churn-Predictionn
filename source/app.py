import streamlit as st
import pandas as pd
import joblib
import os
import sys

# --- YOL AYARLARI ---
current_file_path = os.path.abspath(__file__)
source_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(source_directory)

if source_directory not in sys.path:
    sys.path.append(source_directory)

# --- DOSYA YOLLARI ---
DATA_PATH = os.path.join(project_root, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(project_root, "models", "churn_model.pkl")

# --- IMPORT ---
try:
    from data_prep import load_data, clean_data, encode_data
except ImportError:
    st.error("Hata: data_prep modülü bulunamadı.")
    st.stop()

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Churn Tahmin Sistemi", page_icon="🔮")

# --- BAŞLIK ---
st.title("🔮 Müşteri Terk (Churn) Tahmin Sistemi")
st.markdown("Bu proje, müşterilerin firmayı terk edip etmeyeceğini **Yapay Zeka** ile tahmin eder.")
st.write("---")


# --- MODEL YÜKLEME ---
@st.cache_resource  # Modeli her seferinde tekrar yüklemesin diye önbelleğe alıyoruz
def load_models():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


model = load_models()
if model is None:
    st.error("❌ Model dosyası bulunamadı! Lütfen önce train.py dosyasını çalıştır.")
    st.stop()

# --- YAN MENÜ (KULLANICI GİRİŞİ) ---
st.sidebar.header("Müşteri Bilgileri")


def user_input_features():
    # Kullanıcıdan en önemli 4 veriyi alalım
    contract = st.sidebar.selectbox("Sözleşme Tipi (Contract)", ('Month-to-month', 'One year', 'Two year'))
    tenure = st.sidebar.slider("Abonelik Süresi (Ay)", 1, 72, 12)
    monthly_charges = st.sidebar.slider("Aylık Ücret ($)", 18.0, 120.0, 70.0)
    internet_service = st.sidebar.selectbox("İnternet Servisi", ('DSL', 'Fiber optic', 'No'))

    # Geri kalanlar için varsayılan (dummy) veri oluşturacağız
    # Önce gerçek veriden bir örnek alalım ki sütun isimleri tutsun
    df_raw = load_data(DATA_PATH)
    if df_raw.empty:
        st.error("Veri seti okunamadı.")
        st.stop()

    # Boş bir dataframe şablonu oluştur (tek satırlık)
    input_df = df_raw.iloc[0:1].copy()

    # Kullanıcının seçtiklerini içine yerleştir
    input_df['Contract'] = contract
    input_df['tenure'] = tenure
    input_df['MonthlyCharges'] = monthly_charges
    input_df['InternetService'] = internet_service

    # Churn kolonunu at (çünkü bunu tahmin edeceğiz)
    if 'Churn' in input_df.columns:
        input_df = input_df.drop('Churn', axis=1)

    return input_df, df_raw


# Kullanıcı verisini al
input_df, raw_df = user_input_features()

# --- TAHMİN BUTONU ---
if st.button("TAHMİN ET (Analyze)"):
    # 1. Veri Hazırlığı (Pipeline)
    # Modelin eğitildiği formata getirmemiz lazım

    # Dikkat: Encoding işleminin doğru çalışması için,
    # bizim tek satırlık veriyi, ana veri setine ekleyip encode edip geri alacağız.
    # (Bu basit bir hiledir, encoder'ın tüm seçenekleri görmesi için)

    raw_df_no_target = raw_df.drop('Churn', axis=1)
    combined_df = pd.concat([input_df, raw_df_no_target], axis=0)

    # Temizle ve Encode et
    processed_df = clean_data(combined_df)  # Temizle
    encoded_df = encode_data(processed_df)  # Sayısallaştır

    # Bizim satırımız en baştaki satırdı (index 0)
    final_input = encoded_df.iloc[0:1]

    # 2. Model Tahmini
    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0][1]

    # 3. Sonuç Gösterimi
    st.write("---")
    st.subheader("Sonuç:")

    if prediction == 1:
        st.error(f"🚨 DİKKAT! Bu müşteri **CHURN** edebilir (Gidebilir).")
        st.write(f"Gitme İhtimali: **%{probability * 100:.2f}**")
    else:
        st.success(f"✅ GÜVENLİ. Bu müşteri kalıcı görünüyor.")
        st.write(f"Gitme İhtimali: Sadece **%{probability * 100:.2f}**")

# Alt bilgi
st.write("---")
st.info("Bu proje yapay zeka mentörlüğü kapsamında geliştirilmiştir.")
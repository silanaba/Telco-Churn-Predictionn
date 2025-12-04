import pandas as pd
import joblib
import os
import sys

# --- YOL AYARLARI ---
current_file_path = os.path.abspath(__file__)
source_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(source_directory)

# Modül yolu ekle
if source_directory not in sys.path:
    sys.path.append(source_directory)

# Dosya yolları
DATA_FILE_PATH = os.path.join(project_root, "data", "Telco-Customer-Churn.csv")
MODEL_PATH = os.path.join(project_root, "models", "churn_model.pkl")

# data_prep import
try:
    from data_prep import load_data, clean_data, encode_data
except ImportError:
    print("HATA: data_prep bulunamadı.")
    sys.exit()


def load_trained_model():
    """Kaydedilmiş modeli yükler."""
    if not os.path.exists(MODEL_PATH):
        print("❌ Model dosyası bulunamadı! Önce train.py çalıştırılmalı.")
        return None

    print(f"🧠 Model yükleniyor: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    return model


def make_random_prediction():
    """
    Veriden rastgele bir satır çeker ve modele sorar:
    'Sence bu müşteri Churn eder mi?'
    """
    # 1. Modeli Yükle
    model = load_trained_model()
    if model is None: return

    # 2. Veriyi Hazırla (Modelin tanıdığı hale getir)
    raw_df = load_data(DATA_FILE_PATH)
    clean_df = clean_data(raw_df)
    encoded_df = encode_data(clean_df)

    # 3. Rastgele Bir Müşteri Seç (Test için)
    random_customer = encoded_df.sample(1)

    # Gerçek durumu sakla (Kyaslamak için)
    real_value = random_customer['Churn'].values[0]

    # Tahmin için Churn kolonunu çıkar (Model bunu görmemeli)
    X_input = random_customer.drop('Churn', axis=1)

    # 4. TAHMİN YAP
    prediction = model.predict(X_input)[0]
    probability = model.predict_proba(X_input)[0][1]  # Churn olma ihtimali

    # 5. SONUCU YAZDIR
    print("\n------------------------------------------------")
    print("🔮 TAHMİN SONUCU")
    print("------------------------------------------------")
    print(f"Seçilen Müşteri Özellikleri (Özet):")
    print(X_input.iloc[:, :5].to_string(index=False))  # İlk 5 özelliği göster
    print("...")

    print(f"\nGerçek Durum: {'CHURN (Gitti)' if real_value == 1 else 'KALDI'}")
    print(f"Model Tahmini: {'CHURN (Gider)' if prediction == 1 else 'KALIR'}")
    print(f"Churn İhtimali: %{probability * 100:.2f}")

    if real_value == prediction:
        print("\n✅ DOĞRU BİLDİ!")
    else:
        print("\n❌ YANILDI (Olabilir, %100 başarı imkansızdır)")


if __name__ == "__main__":
    make_random_prediction()
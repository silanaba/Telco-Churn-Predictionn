import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import sys

# ==========================================
# 1. YOL VE KLASÖR AYARLARI
# ==========================================

# Dosyanın olduğu yeri bul (source klasörü)
current_file_path = os.path.abspath(__file__)
source_directory = os.path.dirname(current_file_path)  # Burası 'source' klasörü
project_root = os.path.dirname(source_directory)  # Proje ana dizini

# Python'un data_prep.py dosyasını bulması için 'source' klasörünü sisteme tanıt
if source_directory not in sys.path:
    sys.path.append(source_directory)

# Veri setinin ve Modelin tam yolunu oluştur
DATA_FILE_PATH = os.path.join(project_root, "data", "Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(project_root, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")

# ==========================================
# 2. MODÜLLERİ İÇERİ AL
# ==========================================
try:
    # data_prep dosyasının da 'source' klasöründe olduğundan emin ol!
    from data_prep import load_data, clean_data, encode_data
except ImportError as e:
    print(f"HATA: data_prep modülü bulunamadı!")
    print(f"Lütfen 'data_prep.py' dosyasının '{source_directory}' içinde olduğundan emin ol.")
    print(f"Hata Detayı: {e}")
    sys.exit()


# ==========================================
# 3. FONKSİYONLAR
# ==========================================

def train_model(df: pd.DataFrame):
    """Modeli eğitir."""
    y = df['Churn']
    X = df.drop(['Churn'], axis=1)

    print("✂️  Veri ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 Model eğitiliyor...")
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Model Başarısı (Accuracy): {acc:.4f}")

    return model


def save_model(model):
    """Modeli kaydeder."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"💾 Kaydediliyor: {MODEL_SAVE_PATH}")
    joblib.dump(model, MODEL_SAVE_PATH)

    if os.path.exists(MODEL_SAVE_PATH):
        print("🎉 TEBRİKLER! Model dosyası başarıyla oluşturuldu.")
    else:
        print("❌ Hata: Dosya oluşturulamadı.")


# ==========================================
# 4. ÇALIŞTIRMA
# ==========================================
if __name__ == "__main__":
    print(f"📍 Çalışma Dizini: {project_root}")

    if not os.path.exists(DATA_FILE_PATH):
        print("\n❌ HATA: CSV DOSYASI BULUNAMADI!")
        print(f"Aranan yer: {DATA_FILE_PATH}")
    else:
        raw_df = load_data(DATA_FILE_PATH)
        if not raw_df.empty:
            df = clean_data(raw_df)
            df = encode_data(df)
            model = train_model(df)
            save_model(model)
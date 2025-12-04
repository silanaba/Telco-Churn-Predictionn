import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder


DATA_PATH = "telco_churn/data/Telco-Customer-Churn.csv"
DROP_COLS = ['customerID']


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Belirtilen yoldan CSV dosyasını okur.
    Dosya bulunamazsa hata mesajı basar.
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"HATA: Dosya bulunamadı -> {csv_path}")

        df = pd.read_csv(csv_path)
        print(f"Veri başarıyla yüklendi. Boyut: {df.shape}")
        return df

    except Exception as e:
        print(f"Veri yüklenirken bir hata oluştu: {e}")
        return pd.DataFrame()  # Hata durumunda boş dataframe dön



def check_df(df: pd.DataFrame) -> None:
    """Veri setine genel bakış atar."""
    print("\n##################### SHAPE #####################")
    print(df.shape)
    print("\n##################### INFO #####################")
    print(df.info()) # Info zaten print eder, ekstra print'e gerek yok
    print("\n##################### NA #####################")
    print(df.isnull().sum())
    print("\n##################### HEAD #####################")
    print(df.head())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri temizleme işlemlerini gerçekleştirir:
    1. TotalCharges sayısal tipe çevrilir.
    2. Eksik veriler doldurulur.
    3. Gereksiz kolonlar atılır.
    """
    # Veriyi bozmamak için kopyasını alalım
    df = df.copy()

    # TotalCharges dönüşümü
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Kolon silme
    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        print(f" Silinen kolonlar: {cols_to_drop}")

    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kategorik verileri Label Encoding ile sayısallaştırır.
    Otomatik olarak 'object' tipindeki tüm kolonları bulur.
    """
    df = df.copy()
    le = LabelEncoder()

    # Object tipindeki (yazı) kolonları otomatik bul
    object_cols = df.select_dtypes(include=['object']).columns

    for col in object_cols:
        df[col] = le.fit_transform(df[col])

    print(f" {len(object_cols)} adet kategorik kolon sayısallaştırıldı.")
    return df


if __name__ == "__main__":
    # 1. Veriyi Yükle
    raw_df = load_data(DATA_PATH)

    if not raw_df.empty:
        # 2. Temizle
        clean_df = clean_data(raw_df)

        # 3. Encode Et
        final_df = encode_data(clean_df)

        # Sonucu Gör
        print("\n--- İLK 5 SATIR ---")
        print(final_df.head())
        print("\n--- BİLGİ ---")
        print(final_df.info())




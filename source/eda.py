import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

current_dir = os.path.dirname(os.path.abspath("telco_churn/source/eda.py"))
sys.path.append(current_dir)

from data_prep import load_data, clean_data, DATA_PATH


sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)

def plot_target_distribution(df: pd.DataFrame, target_column: str):
    """
    Hedef değişkenin dağılımını padta grafiği ve sütun grafiği ile gösterir.

    """

    if target_column not in df.columns:
        print(f"{target_column} kolon bulunamadı.")
        return

    fig, axes = plt.subplots(1,2, figsize=(14,6))

    # Pasta Grafiği

    counts = df[target_column].value_counts()
    axes[0].pie(counts, labels = counts.index, autopct='%1.1f%%')
    axes[0].set_title(f"{target_column} Dağılımı")

    # Sütun Grafiği

    sns.countplot(x = target_column, data = df, ax = axes[1])
    axes[1].set_title(f"{target_column} Sayıları")

    plt.tight_layout()
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame):
    """
    Sayısal değişkenlerin histogramlarını çizer.
    """
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Target (eğer sayısalsa) ve gereksizleri çıkarabilirsin

    for col in num_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(df[col], kde=True, color='teal')
        plt.title(f'{col} Dağılımı')
        plt.show()



def plot_correlation_matrix(df: pd.DataFrame):
        """
        Sadece sayısal değişkenler arasındaki ilişkiyi (korelasyonu) gösterir.
        """
        # Sadece sayısal kolonları al
        num_df = df.select_dtypes(include=['float64', 'int64'])

        if num_df.empty:
            print("Sayısal kolon yok, korelasyon çizilemedi.")
            return

        plt.figure(figsize=(12, 10))
        # Korelasyon matrisi
        corr = num_df.corr()

        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
        plt.title("Korelasyon Matrisi")
        plt.show()

if __name__ == "__main__":
        print("Görselleştirme Başlıyor...")

        # 1. Veriyi Hazırla (data_prep'teki fonksiyonları kullanıyoruz)
        raw_df = load_data(DATA_PATH)
        df = clean_data(raw_df)

        # NOT: encode_data fonksiyonunu çağırmadık!
        # Çünkü grafiklerde "0-1" yerine "Yes-No" görmek daha okunaklıdır.

        if not df.empty:
            # 2. Hedef Değişkeni İncele
            plot_target_distribution(df, "Churn")

            # 3. Sayısal Değişkenleri İncele
            plot_numerical_distributions(df)

            # 4. (Opsiyonel) Encode edip Korelasyona Bak
            # Korelasyon için sayısal veri şart, burada encode_data'yı çağırabiliriz veya manuel mapleyebiliriz.
            from data_prep import encode_data
            df_encoded = encode_data(df)
            plot_correlation_matrix(df_encoded)
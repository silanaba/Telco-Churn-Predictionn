# 🔮 Telco Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📌 Proje Hakkında
Bu proje, telekomünikasyon sektöründeki müşterilerin **hizmeti terk etme (churn)** ihtimallerini Yapay Zeka (Makine Öğrenmesi) kullanarak tahmin eder. 

Geliştirilen model, müşteri verilerini (abonelik süresi, ödeme yöntemi, internet servisi vb.) analiz ederek %80 üzeri doğrulukla tahmin yapabilmektedir. Ayrıca **Streamlit** kullanılarak son kullanıcılar için interaktif bir web arayüzü tasarlanmıştır.

## 🚀 Özellikler
- **Veri Analizi:** Eksik veri doldurma, Encoding ve EDA süreçleri.
- **Makine Öğrenmesi:** Random Forest algoritması ile model eğitimi.
- **Model Kayıt:** Eğitilen modelin `.pkl` formatında saklanması.
- **Web Arayüzü:** Kullanıcı dostu Streamlit arayüzü ile anlık tahmin.

## 📂 Proje Yapısı
```text
Telco-Churn-Prediction/
├── data/          # Veri seti (CSV)
├── models/        # Eğitilmiş model dosyası (.pkl)
├── source/        # Kaynak kodlar
│   ├── data_prep.py  # Veri ön işleme
│   ├── train.py      # Model eğitimi
│   ├── app.py        # Streamlit arayüzü
└── README.md



💻 Nasıl Çalıştırılır?
Projeyi klonlayın:

Bash

git clone [https://github.com/silanaba/Telco-Churn-Predictionn.git](https://github.com/silanaba/Telco-Churn-Predictionn.git)
Gerekli kütüphaneleri yükleyin:

Bash

pip install pandas numpy scikit-learn streamlit joblib matplotlib seaborn
Uygulamayı başlatın:

Bash

streamlit run source/app.py
📊 Kullanılan Teknolojiler
Python

Pandas & NumPy (Veri İşleme)

Scikit-learn (Makine Öğrenmesi)

Streamlit (Frontend/Arayüz)

Git & GitHub (Versiyon Kontrolü)

Bu proje Veri Bilimi ve Yapay Zeka alanındaki yetkinlikleri geliştirmek amacıyla hazırlanmıştır.



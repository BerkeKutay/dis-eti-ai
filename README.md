# 🦷 Dental AI — Periodontal Hastalık Tespit Sistemi

> YOLOv8 tabanlı görüntü analizi ve özel eğitilmiş Transformer (LLM) entegrasyonu ile diş eti hastalıklarını tespit eden, klinik değerlendirme raporları üreten yapay zeka sistemi.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-red)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Proje Hakkında

Bu sistem iki ana bileşenden oluşmaktadır:

- **Vision Katmanı (YOLOv8m):** Diş eti fotoğraflarında 7 farklı durumu tespit eder
- **LLM Katmanı (Custom Transformer):** Tespit sonuçlarını klinik dil ile yorumlar

### Tespit Edilen Sınıflar

| Sınıf | Açıklama | Risk |
|-------|----------|------|
| Sağlıklı Doku | Normal diş eti | Düşük |
| Hafif Gingivitis | Erken evre iltihaplanma | Düşük |
| İleri Gingivitis | İleri evre iltihaplanma | Orta |
| Periodontitis | Periodontal kemik kaybı | Yüksek |
| Plak | Bakteriyel biyofilm | Düşük-Orta |
| Tartar (Diş Taşı) | Mineralize plak | Orta |
| Kanama | Gingival kanama | Orta |

### Model Performansı (V5)

| Metrik | V4 (Eski) | V5 (Yeni) | İyileşme |
|--------|-----------|-----------|----------|
| mAP@50 | 0.546 | **0.908** | +%66 |
| Precision | 0.537 | **0.872** | +%62 |
| Recall | 0.651 | **0.879** | +%35 |
| mAP@50-95 | 0.372 | **0.768** | +%106 |

---

## 🚀 Kurulum

### Gereksinimler

- Python 3.10+
- CUDA destekli GPU (önerilir) veya CPU
- 8GB+ RAM

### 1) Repoyu klonla

```bash
git clone https://github.com/BerkeKutay/dis-eti-ai.git
cd dis-eti-ai
```

### 2) Sanal ortam oluştur

```bash
conda create -n dental_gpu python=3.10
conda activate dental_gpu
```

veya venv ile:

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3) Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

---

## 📦 Model Dosyaları

Model ağırlıkları dosya boyutu nedeniyle GitHub'da bulunmamaktadır.  
Aşağıdaki seçeneklerden birini kullan:

### Seçenek A — Hazır modeli indir (Önerilir)

> YOLO modeli ve LLM ağırlıkları için iletişim: [GitHub Issues](https://github.com/BerkeKutay/dis-eti-ai/issues)

Model dosyaları alındıktan sonra şu konumlara yerleştir:

```
dis-eti-ai/
├── runs/
│   └── detect/
│       └── dental_strong_m_v5/
│           └── weights/
│               └── best.pt          ← YOLO model ağırlığı
└── dental_llm_project/
    └── trained_model/
        └── dental_model.pt          ← LLM model ağırlığı
```

### Seçenek B — Kendi modelini eğit

#### YOLO Modeli

```bash
# 1) Dataset'i hazırla (aşağıdaki Dataset Yapısı bölümüne bak)
# 2) Stratified split uygula
python stratified_split.py

# 3) Eğitimi başlat
python train_strong_yolo.py
```

#### LLM Modeli

```bash
cd dental_llm_project

# Tokenizer eğit
python training/train_tokenizer.py

# Model eğit
python training/train_model.py
```

---

## 📁 Dataset Yapısı

Dataset GitHub'da bulunmamaktadır (14GB). Kendi datasetin ile kullanmak için:

```
Dataset/
├── training/
│   ├── images/        ← .jpg, .png görüntüler
│   └── labels/        ← YOLO formatında .txt etiketler
├── validation/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**YOLO Label Formatı:**
```
# class_id  x_center  y_center  width  height  (normalize edilmiş 0-1)
3 0.512 0.489 0.234 0.178
```

**Sınıf ID'leri:**
```
0: saglam
1: hafif_gingivitis
2: ileri_gingivitis
3: periodontitis
4: plak
5: tartar
6: kanama
```

---

## 🖥️ Uygulamayı Çalıştır

Model dosyaları yerleştirildikten sonra:

```bash
python app.py
```

Tarayıcıda aç: `http://127.0.0.1:5000`

---

## 🏗️ Proje Yapısı

```
dis-eti-ai/
├── app.py                          # Flask ana uygulama
├── dataset.yaml                    # YOLO dataset konfigürasyonu
├── train_strong_yolo.py            # YOLO eğitim scripti (oversampling dahil)
├── stratified_split.py             # Dataset stratified split scripti
├── requirements.txt                # Python bağımlılıkları
│
├── dental_llm_project/
│   ├── inference.py                # LLM inference
│   ├── tokenizer/
│   │   ├── vocab.json              # BPE tokenizer vocab
│   │   └── merges.txt              # BPE merge kuralları
│   └── training/
│       ├── model.py                # DentalTransformer mimarisi
│       ├── train_model.py          # LLM eğitim scripti
│       └── train_tokenizer.py      # Tokenizer eğitim scripti
│
├── templates/
│   ├── index.html                  # Ana sayfa
│   └── result.html                 # Analiz sonuç sayfası
│
├── static/
│   ├── style.css
│   ├── results/                    # Analiz görselleri (runtime)
│   └── uploads/                    # Yüklenen görseller (runtime)
│
└── model_analiz_v5.ipynb           # Model performans analizi notebook
```

---

## 🧠 LLM Mimarisi

Foundation model kullanılmamıştır. Tamamen sıfırdan eğitilmiş custom transformer:

```
DentalTransformer
├── dim: 512
├── heads: 8
├── layers: 6
├── max_len: 512
└── vocab_size: 16000
```

50.000 sentetik klinik veri örneği ile eğitilmiştir.

---

## 📊 Sistem Akışı

```
Diş Fotoğrafı
      ↓
YOLOv8m Detection
      ↓
[Sınıf, Confidence, BBox, Konum]
      ↓
Risk Skoru Hesaplama (Ağırlıklı)
      ↓
LLM Klinik Yorum Üretimi
      ↓
Grad-CAM Aktivasyon Haritası
      ↓
Web Arayüzü (Flask)
```

---

## ⚠️ Önemli Not

Bu sistem **yalnızca araştırma amaçlıdır.** Klinik tanı koymak için kullanılamaz. Herhangi bir diş sağlığı problemi için mutlaka bir diş hekimine başvurun.

---

## 📬 İletişim

Sorularınız için [GitHub Issues](https://github.com/BerkeKutay/dis-eti-ai/issues) kullanabilirsiniz.

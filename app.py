import os
import sys
import re
import cv2
import torch
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO

# -------------------------------------------------
# LLM IMPORT
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "dental_llm_project"))
from inference import generate

# -------------------------------------------------
# FLASK
# -------------------------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# YOLO MODEL
# -------------------------------------------------
YOLO_PATH = "runs/detect/dental_strong_m_v5/weights/best.pt"
model = YOLO(YOLO_PATH)
model.model.to(DEVICE)
model.model.eval()

CLASS_NAMES = model.names

# -------------------------------------------------
# RISK MAP
# -------------------------------------------------
RISK_MAP = {
    "hafif_gingivitis": 35,
    "ileri_gingivitis": 60,
    "periodontitis": 85,
    "plak": 40,
    "tartar": 55,
    "kanama": 20
}

# Sinif isimlerinin Turkce karsiliklari (UI'de gosterilir)
SINIF_TURKCE = {
    "hafif_gingivitis":  "Hafif Gingivitis",
    "ileri_gingivitis":  "İleri Gingivitis",
    "periodontitis":     "Periodontitis",
    "plak":              "Plak",
    "tartar":            "Tartar (Diş Taşı)",
    "kanama":            "Kanama",
    "saglam":            "Sağlıklı Doku",
}

# Risk agirlik katsayilari (genel skor hesabi icin)
RISK_AGIRLIK = {
    "periodontitis":    1.0,   # en agir
    "ileri_gingivitis": 0.85,
    "tartar":           0.70,
    "kanama":           0.60,
    "plak":             0.55,
    "hafif_gingivitis": 0.40,
    "saglam":           0.10,
}

# -------------------------------------------------
# KONUM
# -------------------------------------------------
def konum_bul(x1, y1, x2, y2, w, h):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    yatay = "sol" if cx < w/3 else "sağ" if cx > 2*w/3 else "orta"
    dikey = "üst" if cy < h/2 else "alt"
    return f"{yatay}-{dikey}"

# -------------------------------------------------
# GRAD-CAM
# -------------------------------------------------
def generate_gradcam(input_image_path, results):
    img = cv2.imread(input_image_path)
    heatmap = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            heatmap[y1:y2, x1:x2] += 1.0

    if np.max(heatmap) > 0:
        heatmap = cv2.GaussianBlur(heatmap, (51,51), 0)
        heatmap = heatmap / np.max(heatmap)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
    return overlay

# -------------------------------------------------
# RISK HESAP
# -------------------------------------------------
def calculate_overall_risk(analiz_sonuclari):
    """
    Agirlikli risk skoru hesaplar.
    - En yuksek riskli sinif baz alinir
    - Diger siniflar agirlik katsayisiyla katki saglar
    - Sonuc 0-100 araliginda normalize edilir
    """
    if not analiz_sonuclari:
        return 0

    # En yuksek risk skoru
    max_risk = max(x["risk"] for x in analiz_sonuclari)

    # Agirlikli katki: her sinifin risk * agirlik carpimi
    toplam_katki = 0
    toplam_agirlik = 0
    for item in analiz_sonuclari:
        sinif = item["sinif"]
        agirlik = RISK_AGIRLIK.get(sinif, 0.5)
        toplam_katki += item["risk"] * agirlik
        toplam_agirlik += agirlik

    agirlikli_ort = toplam_katki / toplam_agirlik if toplam_agirlik > 0 else 0

    # Max risk ile agirlikli ortalama arasinda dengeleme (60% max, 40% ortalama)
    final_skor = (max_risk * 0.60) + (agirlikli_ort * 0.40)

    return min(round(final_skor), 100)

# -------------------------------------------------
# GENEL RAPOR
# -------------------------------------------------
def generate_general_report(analiz_sonuclari, risk_skoru):
    if risk_skoru >= 80:
        seviye = "yüksek"
    elif risk_skoru >= 50:
        seviye = "orta"
    else:
        seviye = "düşük"

    rapor = f"Toplam periodontal risk skoru {risk_skoru}/100 düzeyindedir. "
    rapor += f"Bu değer {seviye} risk kategorisinde değerlendirilmektedir. "
    rapor += "Klinik bulgular doğrultusunda düzenli periodontal takip önerilmektedir."
    return rapor

# -------------------------------------------------
# LLM ÇIKTI TEMİZLEYİCİ  ← TAMAMEN YENİ
# -------------------------------------------------
def turkce_capitalize(s):
    """Python upper() Turkce I/i duzeltmesi yapmaz, bu fonksiyon yapar."""
    if not s:
        return s
    first = s[0]
    rest = s[1:]
    tr_map = {'i': 'İ', 'ı': 'I'}
    return tr_map.get(first, first.upper()) + rest

def clean_llm_output(raw_text, max_evaluation_sentences=3, max_suggestions=2):
    """
    LLM'den gelen ham metni yapılandırır:
    - Prompt kalıntılarını (Klinik durum:, Risk:, Lokalizasyon:) siler
    - Tekrar eden cümleleri kaldırır
    - Değerlendirme kısmını max_evaluation_sentences cümleyle sınırlar
    - Öneri kısmını max_suggestions öneriyle sınırlar
    - Temiz bir dict döner: {"degerlendirme": str, "oneriler": [str, str]}
    """

    # 1) Prompt satırlarını ve boşlukları temizle
    lines = raw_text.split("\n")
    skip_prefixes = (
        "klinik durum:", "risk seviyesi:", "risk:", "lokalizasyon:",
        "Klinik durum:", "Risk seviyesi:", "Risk:", "Lokalizasyon:"
    )
    lines = [l.strip() for l in lines if l.strip() and not l.strip().lower().startswith(skip_prefixes)]

    # 2) Tüm metni birleştir
    full_text = " ".join(lines)

    # --- Gurultu temizleme ---

    # "- Oneri:", "Oneri:" etiketlerini kaldir
    full_text = re.sub(r'-?\s*[Öö]neri\s*:', '', full_text)

    # Tum inline etiketleri kaldir (tire'li ve tiresiz)
    full_text = re.sub(
        r'-?\s*(Klinik durum|Risk seviyesi|Risk|Değerlendirme|Lokalizasyon)\s*:',
        '', full_text, flags=re.IGNORECASE
    )

    # Lokalizasyon token artiklari - tire'li: "sag-", "sol-"
    full_text = re.sub(r'\b(sağ|sol|orta|üst|alt)\s*-\s*', '', full_text)

    # Lokalizasyon token artiklari - tiresiz yalniz duran
    full_text = re.sub(r'(?<![a-zA-ZğüşıöçĞÜŞİÖÇ])(sağ|sol|orta|üst|alt)(?![a-zA-ZğüşıöçĞÜŞİÖÇ])', '', full_text, flags=re.IGNORECASE)

    # Risk seviyesi artiklari - yalniz duran: "düşük", "orta", "yüksek"
    full_text = re.sub(r'(?<![a-zA-ZğüşıöçĞÜŞİÖÇ])(düşük|orta|yüksek)(?![a-zA-ZğüşıöçĞÜŞİÖÇ])', '', full_text)

    # "X olarak degerlendirilmistir" - baslangici kopuk kalan meta cumle kalintilari
    full_text = re.sub(r'[a-zA-ZğüşıöçĞÜŞİÖÇ\s]+olarak değerlendirilmiştir\.?\s*', '', full_text)

    # Oneri baglaclari: "Doku yanıtı incelendiğinde X", "Periodontal muayene kapsamında X"
    # gibi LLM'in oneri basina ekledigi baglac cumleciklerini kaldir
    # Pattern: nokta/virgul olmadan uzun bir ifade + büyük harf ile baslayan asil oneri
    full_text = re.sub(
        r'(Doku yanıtı incelendiğinde|Periodontal muayene kapsamında|'
        r'Lokal doku değerlendirmesinde|Klinik gözlem verilerine göre|'
        r'Bölgesel değerlendirme kapsamında|Ağız içi analiz sırasında|'
        r'Anterior bölgesinde|Posterior bölgesinde|Bölgesinde|'
        r'İncelendiğinde|Değerlendirildiğinde|Klinik olarak)\s+',
        '', full_text, flags=re.IGNORECASE
    )

    # Kopuk meta cumlecikler: "Gozlenmistir.", "Mevcuttur." gibi tek basina anlamsiz cumleler
    # (nokta ile biten, 1-2 kelimelik, bilgi icermeyen)
    full_text = re.sub(r'(?<![.!?])[A-ZİIÜÖĞŞÇa-zışüöğç]+(miştir|mıştır|muştur|müştür|miştir)\.', '', full_text)

    # Bastaki tire/nokta/bosluk artiklarini temizle
    full_text = re.sub(r'^[\s\-\.]+', '', full_text)

    # Birden fazla boslugu tek bosluga indir
    full_text = re.sub(r'\s{2,}', ' ', full_text).strip()

    # Cümle bölücü
    raw_sentences = re.split(r'(?<=[.!?])\s+', full_text)

    # 3) Tekrar eden cümleleri kaldır (normalize ederek karşılaştır)
    seen_normalized = set()
    unique_sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        normalized = re.sub(r'\s+', ' ', s.lower().strip())
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            unique_sentences.append(s)

    # 4) Benzer cümleleri de filtrele (ilk 10 kelime aynıysa tekrar sayılır)
    final_sentences = []
    seen_starts = set()
    for s in unique_sentences:
        start = " ".join(s.lower().split()[:8])
        if start not in seen_starts:
            seen_starts.add(start)
            final_sentences.append(s)

    # 5) Önerileri ve değerlendirmeleri ayır
    oneri_keywords = ("öneri", "önerilir", "önerilmektedir", "planlanmalı",
                      "uygulanmalı", "gerektirir", "geciktirilmemeli",
                      "yeterlidir", "yeterli olabilir")

    degerlendirme_cumleleri = []
    oneri_cumleleri = []

    for s in final_sentences:
        s_lower = s.lower()
        if any(kw in s_lower for kw in oneri_keywords):
            oneri_cumleleri.append(s)
        else:
            degerlendirme_cumleleri.append(s)

    # 6) Oneri cumlelerini baglactan temizle
    baglac_pattern = re.compile(
        r'^(Doku yanıtı incelendiğinde|Periodontal muayene kapsamında|'
        r'Lokal doku değerlendirmesinde|Klinik gözlem verilerine göre|'
        r'Bölgesel değerlendirme kapsamında|İncelendiğinde|Değerlendirildiğinde|'
        r'Bölgede|Klinik olarak)\s+',
        re.IGNORECASE
    )
    oneri_cumleleri = [baglac_pattern.sub('', s).strip() for s in oneri_cumleleri]
    oneri_cumleleri = [turkce_capitalize(s) for s in oneri_cumleleri]

    # 7) Degerlendirme cumlelerini buyuk harfle baslatarak birlestir
    degerlendirme_cumleleri = [turkce_capitalize(s) for s in degerlendirme_cumleleri]
    degerlendirme = " ".join(degerlendirme_cumleleri[:max_evaluation_sentences])
    oneriler = oneri_cumleleri[:max_suggestions]

    # Edge case: hiç değerlendirme yoksa ilk cümleleri kullan
    if not degerlendirme and final_sentences:
        degerlendirme = " ".join(final_sentences[:max_evaluation_sentences])

    # Edge case: hiç öneri yoksa son cümleyi öneri say
    if not oneriler and len(final_sentences) > max_evaluation_sentences:
        oneriler = [final_sentences[max_evaluation_sentences]]

    return {
        "degerlendirme": degerlendirme.strip(),
        "oneriler": [o.strip() for o in oneriler]
    }


# -------------------------------------------------
# ROUTE
# -------------------------------------------------
@app.route("/", methods=["GET","POST"])
def index():

    if request.method == "POST":

        file = request.files.get("image")
        if not file:
            return render_template("index.html")

        filename = file.filename
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)

        img = cv2.imread(upload_path)
        h, w = img.shape[:2]

        results = model(upload_path)

        analiz_dict = {}

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0]) * 100
                sinif = CLASS_NAMES.get(cls, "bilinmeyen")

                if sinif not in RISK_MAP:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                konum = konum_bul(x1,y1,x2,y2,w,h)

                if sinif not in analiz_dict or conf > analiz_dict[sinif]["oran"]:
                    analiz_dict[sinif] = {
                        "sinif": sinif,
                        "sinif_turkce": SINIF_TURKCE.get(sinif, sinif),
                        "oran": round(conf,1),
                        "risk": RISK_MAP[sinif],
                        "konum": konum,
                        "bbox": (x1,y1,x2,y2)
                    }

        analiz_sonuclari = list(analiz_dict.values())

        if not analiz_sonuclari:
            return render_template("index.html", error_message="Diş eti algılanamadı.")

        # -------------------------------------------------
        # LLM — temizlenmiş yapılandırılmış çıktı
        # -------------------------------------------------
        for item in analiz_sonuclari:

            risk_text = (
                "yüksek" if item["risk"] > 70 else
                "orta" if item["risk"] > 40 else
                "düşük"
            )

            prompt = f"""
Klinik durum: {item['sinif']}
Risk seviyesi: {risk_text}
Lokalizasyon: {item['konum']}
"""
            raw = generate(prompt)

            # Yapılandırılmış temiz çıktı: {"degerlendirme": str, "oneriler": [str, str]}
            item["yorum"] = clean_llm_output(raw)

        risk_skoru = calculate_overall_risk(analiz_sonuclari)
        genel_yorum = generate_general_report(analiz_sonuclari, risk_skoru)

        # -------------------------------------------------
        # GÖRSELLER
        # -------------------------------------------------
        clean_img = img.copy()
        for item in analiz_sonuclari:
            x1,y1,x2,y2 = item["bbox"]
            cv2.rectangle(clean_img,(x1,y1),(x2,y2),(0,255,0),2)

        boxed_filename = "boxed_" + filename
        cv2.imwrite(os.path.join(RESULT_FOLDER, boxed_filename), clean_img)

        gradcam_img = generate_gradcam(upload_path, results)
        gradcam_filename = "gradcam_" + filename
        cv2.imwrite(os.path.join(RESULT_FOLDER, gradcam_filename), gradcam_img)

        return render_template(
            "result.html",
            results=analiz_sonuclari,
            risk_skoru=risk_skoru,
            genel_yorum=genel_yorum,
            boxed_image=boxed_filename,
            gradcam_image=gradcam_filename
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
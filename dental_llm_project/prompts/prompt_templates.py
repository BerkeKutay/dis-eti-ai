# prompts/prompt_templates.py

periodontitis_prompts = [
"""
Aşağıdaki periodontal bulguları klinik açıdan değerlendir:

- Tanı: Periodontitis
- Enflamasyon şiddeti: {severity}
- Yayılım: {extent}
- Kanama: {bleeding}
- Plak varlığı: {plaque}
- Model güven skoru: {confidence}

Profesyonel bir periodontal değerlendirme paragrafı yaz.
Risk seviyesini belirt ve kısa tedavi önerisi ekle.
""",

"""
Bir periodontoloji uzmanı olarak aşağıdaki vakayı yorumla:

- Tanı: Periodontitis
- Şiddet: {severity}
- Yayılım: {extent}
- Kanama: {bleeding}
- Plak: {plaque}
- Güven: {confidence}

Hastalığın evresini, ilerleme riskini ve önerilen periodontal müdahaleyi açıkla.
"""
]

gingivitis_prompts = [
"""
Tanı: İleri gingivitis
Enflamasyon: {severity}
Kanama: {bleeding}
Plak: {plaque}
Yayılım: {extent}
Güven: {confidence}

Klinik rapor formatında değerlendirme yaz.
""",

"""
Hafif gingivitis ile uyumlu bulgular mevcut.

- Şiddet: {severity}
- Kanama: {bleeding}
- Plak: {plaque}
- Yayılım: {extent}
- Güven: {confidence}

Erken müdahale açısından klinik öneri yaz.
"""
]

tartar_prompts = [
"""
Diş yüzeylerinde tartar birikimi saptanmıştır.

- Yayılım: {extent}
- Enflamasyon: {severity}
- Güven: {confidence}

Klinik etkilerini ve olası periodontal sonuçlarını açıkla.
"""
]

healthy_prompts = [
"""
Periodontal açıdan sağlıklı görünen bir hastanın değerlendirmesini yap.

- Güven: {confidence}

Klinik yorum ve koruyucu öneri yaz.
"""
]
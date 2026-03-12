import os
import shutil
import random
import torch
from pathlib import Path
from ultralytics import YOLO

# -------------------------------------------------
# OVERSAMPLING
# ileri_gingivitis (class 2) ve periodontitis (class 3)
# az örnekli — train setine kopyalayarak çoğalt
# -------------------------------------------------

MINORITY_CLASSES = {2, 3}   # ileri_gingivitis, periodontitis
OVERSAMPLE_TIMES = 4         # her görüntüyü 4x kopyala

def oversample_minority(train_images_dir, train_labels_dir):
    """
    Minority sınıf içeren görüntüleri tespit edip
    train klasörüne kopyalar (OVERSAMPLE_TIMES kadar).
    Orijinal dosyalara dokunmaz.
    """
    images_dir = Path(train_images_dir)
    labels_dir = Path(train_labels_dir)

    label_files = list(labels_dir.glob("*.txt"))
    minority_pairs = []

    for lf in label_files:
        try:
            lines = lf.read_text().strip().splitlines()
            classes_in_file = {int(l.split()[0]) for l in lines if l.strip()}
            if classes_in_file & MINORITY_CLASSES:
                for ext in [".jpg", ".jpeg", ".png"]:
                    img = images_dir / (lf.stem + ext)
                    if img.exists():
                        minority_pairs.append((img, lf))
                        break
        except Exception:
            continue

    print(f"[Oversample] {len(minority_pairs)} minority goruntu bulundu.")

    created = 0
    for img_path, lbl_path in minority_pairs:
        for i in range(1, OVERSAMPLE_TIMES + 1):
            new_stem = f"{img_path.stem}_os{i}"
            new_img = images_dir / (new_stem + img_path.suffix)
            new_lbl = labels_dir / (new_stem + ".txt")
            if not new_img.exists():
                shutil.copy2(img_path, new_img)
                shutil.copy2(lbl_path, new_lbl)
                created += 1

    print(f"[Oversample] {created} yeni dosya olusturuldu.")


def cleanup_oversampled(train_images_dir, train_labels_dir):
    """
    Egitim bittikten sonra _os* ile biten kopyalari sil.
    """
    for d in [train_images_dir, train_labels_dir]:
        for f in Path(d).glob("*_os*"):
            f.unlink()
    print("[Cleanup] Oversample dosyalari silindi.")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    print("======== GPU KONTROL ========")
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("=============================")

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_YAML = os.path.join(BASE_DIR, "dataset.yaml")

    TRAIN_IMAGES = os.path.join(BASE_DIR, "Dataset/training/images")
    TRAIN_LABELS = os.path.join(BASE_DIR, "Dataset/training/labels")

    # Oversampling uygula
    oversample_minority(TRAIN_IMAGES, TRAIN_LABELS)

    try:
        model = YOLO("yolov8m.pt")

        model.train(
            data=DATA_YAML,

            # Core
            epochs=120,
            imgsz=640,
            batch=6,
            workers=4,
            device=0,
            amp=True,

            # Optimizer
            optimizer="AdamW",
            lr0=0.001,
            weight_decay=0.0005,
            cos_lr=True,

            # Warmup
            warmup_epochs=3,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,

            # Augmentation — minority siniflara daha iyi coverage icin arttirildi
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            degrees=10,        # onceki: 5
            translate=0.15,    # onceki: 0.1
            scale=0.5,         # onceki: 0.4
            shear=3,           # onceki: 2
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.8,        # onceki: 0.3 — az ornekli siniflar mosaic ile daha cok gorunur
            mixup=0.1,         # onceki: 0.05
            copy_paste=0.1,    # YENİ — az ornekli bbox'lari baska goruntulere yapistir

            # Loss agirliklari
            cls=1.5,           # onceki: 1.0 — sinif kaybi daha onemli
            box=7.5,
            dfl=1.5,

            # Early stopping
            patience=30,

            # Run klasoru — yeni versiyon
            project="runs/detect",
            name="dental_strong_m_v5",
            exist_ok=True
        )

    finally:
        # Egitim bitince (hata olsa da) kopyalari temizle
        cleanup_oversampled(TRAIN_IMAGES, TRAIN_LABELS)


if __name__ == "__main__":
    main()
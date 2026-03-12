"""
Stratified Dataset Split
------------------------
Mevcut Dataset klasöründeki tüm görüntüleri sınıf dağılımını
koruyarak training / validation / test olarak yeniden böler.

Çalıştırmadan önce:
    python stratified_split.py --dry-run   # sadece istatistik göster
Gerçek split için:
    python stratified_split.py
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# -------------------------------------------------
# AYARLAR
# -------------------------------------------------
BASE_DIR     = Path("Dataset")
SPLITS       = ["training", "validation", "test"]
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
TEST_RATIO   = 0.10
RANDOM_SEED  = 42

CLASS_NAMES = {
    0: "saglam",
    1: "hafif_gingivitis",
    2: "ileri_gingivitis",
    3: "periodontitis",
    4: "plak",
    5: "tartar",
    6: "kanama",
}

# -------------------------------------------------
# YARDIMCI FONKSİYONLAR
# -------------------------------------------------

def get_dominant_class(label_path):
    """
    Bir label dosyasındaki en çok tekrar eden sınıfı döner.
    Stratified split için "bu görüntü hangi sınıfa ait" sorusunu cevaplar.
    """
    counts = defaultdict(int)
    try:
        for line in label_path.read_text().strip().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    except Exception:
        return -1
    return max(counts, key=counts.get) if counts else -1


def collect_all_pairs():
    """
    Tüm split'lerdeki (image, label) çiftlerini toplar.
    Backup varsa oradan okur (daha güvenli).
    _os* ile biten oversample kopyalarını atlar.
    """
    # Backup varsa oradan oku — clear_splits sonrası kayıp olmaz
    source = Path("Dataset_backup") if Path("Dataset_backup").exists() else BASE_DIR

    pairs = []
    for split in SPLITS:
        img_dir = source / split / "images"
        lbl_dir = source / split / "labels"

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if "_os" in img_path.stem:
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))

    return pairs


def stratified_split(pairs):
    """
    Dominant sınıfa göre gruplara ayır, her gruptan
    TRAIN/VAL/TEST oranında örnekler seç.
    """
    random.seed(RANDOM_SEED)

    # Sınıfa göre grupla
    class_groups = defaultdict(list)
    for img_path, lbl_path in pairs:
        cls = get_dominant_class(lbl_path)
        class_groups[cls].append((img_path, lbl_path))

    train_set, val_set, test_set = [], [], []

    for cls, items in class_groups.items():
        random.shuffle(items)
        n = len(items)
        n_val  = max(1, int(n * VAL_RATIO))
        n_test = max(1, int(n * TEST_RATIO))
        n_train = max(0, n - n_val - n_test)
        # Az ornekli siniflar icin: en az 1 val ve 1 test yeterli
        if n_train == 0 and n >= 3:
            n_val, n_test = 1, 1
            n_train = n - 2
        elif n < 3:
            # Cok az ornek: hepsini train'e koy
            n_train, n_val, n_test = n, 0, 0

        train_set.extend(items[:n_train])
        val_set.extend(items[n_train:n_train + n_val])
        test_set.extend(items[n_train + n_val:])

        name = CLASS_NAMES.get(cls, f"class_{cls}")
        print(f"  {name:25s}: toplam={n:4d}  train={n_train:4d}  val={n_val:3d}  test={n_test:3d}")

    return train_set, val_set, test_set


def copy_pairs_to_temp(pairs, tmp_dir):
    """Önce geçici klasöre kopyala."""
    (tmp_dir / "images").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "labels").mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        dst_img = tmp_dir / "images" / img_path.name
        dst_lbl = tmp_dir / "labels" / lbl_path.name
        if not dst_img.exists():
            shutil.copy2(img_path, dst_img)
        if not dst_lbl.exists():
            shutil.copy2(lbl_path, dst_lbl)


def copy_pairs(pairs, split_name, dry_run=False):
    img_dir = BASE_DIR / split_name / "images"
    lbl_dir = BASE_DIR / split_name / "labels"

    if not dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in pairs:
        dst_img = img_dir / img_path.name
        dst_lbl = lbl_dir / lbl_path.name
        if not dry_run:
            shutil.copy2(img_path, dst_img)
            shutil.copy2(lbl_path, dst_lbl)


def backup_dataset():
    backup = Path("Dataset_backup")
    if backup.exists():
        print(f"[Backup] Dataset_backup zaten var, atlanıyor.")
        return
    print(f"[Backup] Dataset klasörü yedekleniyor → Dataset_backup ...")
    shutil.copytree(BASE_DIR, backup)
    print(f"[Backup] Tamamlandı.")


def clear_splits():
    """Mevcut split klasörlerini temizle (sadece images ve labels)."""
    for split in SPLITS:
        for sub in ["images", "labels"]:
            d = BASE_DIR / split / sub
            if d.exists():
                for f in d.iterdir():
                    if "_os" not in f.stem:
                        f.unlink()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Dosya kopyalamadan sadece istatistik göster")
    args = parser.parse_args()

    print("\n=== Stratified Dataset Split ===\n")

    # 1) Tüm çiftleri topla
    all_pairs = collect_all_pairs()
    print(f"Toplam görüntü: {len(all_pairs)}\n")

    # 2) Stratified split
    print("Sınıf bazlı dağılım:")
    train_set, val_set, test_set = stratified_split(all_pairs)

    print(f"\nSonuç:")
    print(f"  Training  : {len(train_set)}")
    print(f"  Validation: {len(val_set)}")
    print(f"  Test      : {len(test_set)}")

    if args.dry_run:
        print("\n[DRY RUN] Dosya kopyalanmadı. Gerçek split için: python stratified_split.py")
        return

    # 3) Yedek al
    backup_dataset()

    # 4) Once gecici klasore kopyala (kaynak = hedef sorununu onlemek icin)
    tmp = Path("_split_tmp")
    print("\n[Split] Gecici klasore kopyalanıyor...")
    copy_pairs_to_temp(train_set, tmp / "training")
    copy_pairs_to_temp(val_set,   tmp / "validation")
    copy_pairs_to_temp(test_set,  tmp / "test")

    # 5) Mevcut splitleri temizle
    print("[Split] Mevcut dosyalar temizleniyor...")
    clear_splits()

    # 6) Gecici klasorden asil konuma tasi
    print("[Split] Dosyalar asil konuma tasınıyor...")
    for split in SPLITS:
        for sub in ["images", "labels"]:
            src = tmp / split / sub
            dst = BASE_DIR / split / sub
            dst.mkdir(parents=True, exist_ok=True)
            if src.exists():
                for f in src.iterdir():
                    shutil.move(str(f), dst / f.name)

    # 7) Gecici klasoru sil
    shutil.rmtree(tmp, ignore_errors=True)

    print("\n✅ Stratified split tamamlandı!")
    print("   Orijinal dataset: Dataset_backup/")
    print("   Yeni dataset    : Dataset/")
    print("\nSonraki adım: python train_strong_yolo.py")


if __name__ == "__main__":
    main()
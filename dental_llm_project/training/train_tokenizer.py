from tokenizers import ByteLevelBPETokenizer
import os

DATA_FILE = "dataset/dental_multi_style_dataset.jsonl"
OUTPUT_DIR = "tokenizer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset'ten sadece metni çıkar
def extract_texts():
    texts = []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            import json
            entry = json.loads(line)
            texts.append(entry["input"])
            texts.append(entry["output"])
    return texts

texts = extract_texts()

# Geçici txt dosyası oluştur
with open("tokenizer_corpus.txt", "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t + "\n")

# ByteLevel BPE eğitimi
tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=["tokenizer_corpus.txt"],
    vocab_size=16000,
    min_frequency=2,
    special_tokens=["<s>", "<pad>", "</s>", "<unk>"]
)

tokenizer.save_model(OUTPUT_DIR)

print("Tokenizer eğitildi ve kaydedildi.")
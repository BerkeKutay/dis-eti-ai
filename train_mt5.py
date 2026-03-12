import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # GPU kapat
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from datasets import load_dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

print("Model yükleniyor...")

model_name = "google/mt5-small"

tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cpu")
model.to(device)

print("Dataset yükleniyor...")

dataset = load_dataset("csv", data_files="clinical_dataset_v3.csv")

def preprocess(example):

    inputs = tokenizer(
        example["input"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

    targets = tokenizer(
        example["output"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = targets["input_ids"]

    # 🔥 PADDING MASK FIX (EN KRİTİK KISIM)
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    inputs["labels"] = labels
    return inputs


dataset = dataset["train"].map(preprocess, batched=True)

print("Training başlıyor...")

training_args = TrainingArguments(
    output_dir="./mt5_dental_model",
    per_device_train_batch_size=1,  # 🔥 MAC için güvenli
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=False,
    dataloader_num_workers=0,
    no_cuda=True  # 🔥 GPU tamamen kapalı
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

print("Model kaydediliyor...")

model.save_pretrained("./mt5_dental_model")
tokenizer.save_pretrained("./mt5_dental_model")

print("✅ Eğitim tamamlandı ve ./mt5_dental_model klasörüne kaydedildi.")

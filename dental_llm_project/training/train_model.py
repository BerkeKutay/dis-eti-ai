import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
import json
from model import DentalTransformer
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4   
MAX_LEN = 256
PAD_TOKEN_ID = 0  # tokenizer special token <pad> indexi 0 olacak

# -------------------------------------------------
# TOKENIZER LOAD
# -------------------------------------------------

tokenizer = ByteLevelBPETokenizer(
    "tokenizer/vocab.json",
    "tokenizer/merges.txt"
)

# -------------------------------------------------
# DATASET
# -------------------------------------------------

class DentalDataset(Dataset):
    def __init__(self, path):
        self.samples = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                text = entry["input"] + " " + entry["output"]

                ids = tokenizer.encode(text).ids
                ids = ids[:MAX_LEN]

                if len(ids) > 1:
                    self.samples.append(ids)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


# -------------------------------------------------
# DYNAMIC PADDING COLLATE
# -------------------------------------------------

def collate_fn(batch):

    max_len = max(len(x) for x in batch)

    padded_batch = []

    for x in batch:
        pad_len = max_len - len(x)
        if pad_len > 0:
            padding = torch.full((pad_len,), PAD_TOKEN_ID, dtype=torch.long)
            x = torch.cat([x, padding])
        padded_batch.append(x)

    batch = torch.stack(padded_batch)

    # Causal LM
    x = batch[:, :-1]
    y = batch[:, 1:]

    return x, y


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

dataset = DentalDataset("dataset/dental_multi_style_dataset.jsonl")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# -------------------------------------------------
# MODEL
# -------------------------------------------------

model = DentalTransformer().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=0.01
)

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN_ID,
    label_smoothing=0.1
)

# -------------------------------------------------
# TRAIN LOOP
# -------------------------------------------------

for epoch in range(EPOCHS):

    model.train()
    total_loss = 0

    for step, (x, y) in enumerate(loader):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss {loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    print(f"\nEpoch {epoch+1} Finished | Avg Loss: {avg_loss:.4f}\n")


# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

os.makedirs("trained_model", exist_ok=True)

torch.save(model.state_dict(), "trained_model/dental_model.pt")

print("Model eğitimi tamamlandı ve kaydedildi.")
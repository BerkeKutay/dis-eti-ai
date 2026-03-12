import os
import torch
from training.model import DentalTransformer
from tokenizers import ByteLevelBPETokenizer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = ByteLevelBPETokenizer(
    os.path.join(BASE_DIR, "tokenizer/vocab.json"),
    os.path.join(BASE_DIR, "tokenizer/merges.txt")
)

model = DentalTransformer()
model.load_state_dict(
    torch.load(
        os.path.join(BASE_DIR, "trained_model/dental_model.pt"),
        map_location=DEVICE
    )
)
model.to(DEVICE)
model.eval()


def generate(
    prompt,
    max_new_tokens=120,
    temperature=0.9,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1
):

    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_new_tokens):

        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # repetition penalty (stabil versiyon)
        if repetition_penalty != 1.0:
            for token_id in torch.unique(input_ids):
                next_token_logits[:, token_id] /= repetition_penalty

        next_token_logits = next_token_logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)

        # top-k
        if top_k > 0:
            topk_probs, topk_indices = torch.topk(probs, top_k)
            probs_filtered = torch.zeros_like(probs)
            probs_filtered.scatter_(1, topk_indices, topk_probs)
            probs = probs_filtered
            probs = probs / probs.sum(dim=-1, keepdim=True)

        # top-p (güvenli versiyon)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            cutoff = cumulative_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False

            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(-1, next_token)

        else:
            next_token = torch.multinomial(probs, 1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if input_ids.shape[1] > 300:
            break

    return tokenizer.decode(input_ids[0].tolist())


test_prompt = "Klinik durum: periodontal dokuları etkileyen inflamatuar süreç | Risk seviyesi: düşük | Lokalizasyon: sağ-üst"

print(generate(test_prompt))
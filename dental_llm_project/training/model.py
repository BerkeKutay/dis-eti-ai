import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, causal_mask):
        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=causal_mask
        )

        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class DentalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size=16000,
        dim=768,          # Büyüttük
        heads=12,         # Daha güçlü attention
        layers=8,         # Daha derin model
        max_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.dim = dim
        self.max_len = max_len

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dropout)
            for _ in range(layers)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def generate_causal_mask(self, T, device):
        mask = torch.triu(
            torch.ones(T, T, device=device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, input_ids):
        B, T = input_ids.shape

        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        causal_mask = self.generate_causal_mask(T, input_ids.device)

        for layer in self.layers:
            x = layer(x, causal_mask)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import torch

model_path = "./mt5_dental_model"

print("Model yükleniyor...")

tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)
model.eval()

print("Model yüklendi.")

test_input = "Tanı: hafif_gingivitis | Olasılık: 72 | Konum: orta-alt | Risk: 35"

inputs = tokenizer(test_input, return_tensors="pt")

print("Generate başlıyor...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=200,
        num_beams=4,
        no_repeat_ngram_size=3
    )

print("Generate bitti.")

decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("ÇIKTI:")
print(decoded)

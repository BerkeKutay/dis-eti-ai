from huggingface_hub import hf_hub_download
import os

os.makedirs("runs/detect/dental_strong_m_v5/weights", exist_ok=True)
os.makedirs("dental_llm_project/trained_model", exist_ok=True)

print("YOLO modeli indiriliyor...")
hf_hub_download(
    repo_id="Kutay0/dis-eti-ai",
    filename="yolo/best.pt",
    local_dir=".",
    repo_type="model"
)

print("LLM modeli indiriliyor...")
hf_hub_download(
    repo_id="Kutay0/dis-eti-ai",
    filename="llm/dental_model.pt",
    local_dir=".",
    repo_type="model"
)

print("Modeller indirildi!")

import torch
import os
import timm 
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("--- Starting Model Caching ---")

print("Caching MiDaS model...")
try:
    os.environ['TORCH_HOME'] = os.environ.get('TORCH_HOME', '/app/torch_cache')
    
    torch.hub.load("intel-isl/MiDaS", "MiDaS_small", force_reload=True)
    torch.hub.load("intel-isl/MiDaS", "transforms", force_reload=True)
    print("MiDaS cached successfully.")
except Exception as e:
    print(f"Failed to cache MiDaS: {e}")

print("Caching t5-small text model...")
try:
    MODEL_ID = "google-t5/t5-small"
    SAVE_PATH = "./t5-small-local"
    os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/app/huggingface_cache')
    
    print(f"Downloading from {MODEL_ID} and saving to {SAVE_PATH}...")
    
    # Download and cache tokenizer, then save to our local path
    tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(SAVE_PATH)
    
    # Download and cache model, then save to our local path
    model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
    model.save_pretrained(SAVE_PATH)

    print(f"t5-small cached successfully to {SAVE_PATH}.")
except Exception as e:
    print(f"Failed to cache t5-small: {e}")


print("--- Model Caching Complete ---")
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Model cache
llama_model = None
llama_tokenizer = None

# Optional: set a custom cache directory if needed
custom_cache_dir = "/cluster/scratch/gcardenal/LLM_models"

def load_llama_8b_model():
    """
    Loads the local LLaMA-8B model if not already loaded.
    """
    global llama_model, llama_tokenizer

    if llama_model is None or llama_tokenizer is None:
        print("Loading LLaMA-8B model locally...")
        llama_tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct", 
            cache_dir=custom_cache_dir
        )
        
        # Fix: set pad token if missing
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token  # or '[PAD]' if you prefer to add a new one

        llama_model = AutoModel.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            cache_dir=custom_cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        llama_model.eval()

def get_llama_8b_embedding(text):
    """
    Computes the embedding of the input text using the last hidden state (mean pooled).
    """
    load_llama_8b_model()

    # Tokenize and send to same device as model
    inputs = llama_tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(llama_model.device)

    with torch.no_grad():
        outputs = llama_model(**inputs, output_hidden_states=True)

    last_hidden = outputs.hidden_states[-1]  # shape: (1, seq_len, hidden_dim)

    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_embeddings = last_hidden * attention_mask
    sum_embeddings = masked_embeddings.sum(dim=1)
    lengths = attention_mask.sum(dim=1).clamp(min=1e-9)
    mean_pooled = sum_embeddings / lengths

    return mean_pooled.squeeze(0).cpu().numpy()  # shape: (hidden_dim,)

# Load data
with open("patient_summaries_top28_val.json") as f:
    data = json.load(f)

# Compute embeddings
embeddings = []

for idx, item in enumerate(data):
    text = item['summary_text']
    print(f"Processing item {idx + 1}/{len(data)}...")

    try:
        embedding = get_llama_8b_embedding(text)
        embeddings.append(embedding)
        print(embedding)
        print(" → Success. Shape:", embedding.shape)
    except Exception as e:
        print(" → Failed to embed:", e)
        embeddings.append(np.zeros((4096,)))  # fallback vector

# Convert to NumPy array
embedding_array = np.vstack(embeddings)

# Save as .npy
np.save("/cluster/scratch/gcardenal/llama_input_val_embeddings.npy", embedding_array)

# Save as CSV
pd.DataFrame(embedding_array).to_csv("/cluster/scratch/gcardenal/llama_input_val_embeddings.csv", index=False)

print("Embedding extraction complete. Saved to llama_input_embeddings.npy and .csv")

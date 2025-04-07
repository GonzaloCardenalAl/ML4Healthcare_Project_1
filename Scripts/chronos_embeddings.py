import pandas as pd  # requires: pip install pandas
import torch
from chronos import BaseChronosPipeline
from tqdm import tqdm
import numpy as np
import os
# Custom cache path (change this to a location with more space!)
os.environ['HF_HOME'] = '/cluster/scratch/gcardenal/.cache/'
os.environ['TRANSFORMERS_CACHE'] = '/cluster/scratch/gcardenal/.cache/'

pipeline = BaseChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",  # use "amazon/chronos-bolt-small" for the corresponding Chronos-Bolt model
    device_map="cuda",  # use "cpu" for CPU inference
    torch_dtype=torch.bfloat16,
)


def extract_patient_embeddings(
    file_path: str,
    pipeline,
    n_patients: int = 4000,
    time_steps: int = 49,
    save_prefix: str = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts per-channel patient embeddings from a multivariate time-series CSV file
    using a Chronos univariate pipeline encoder.

    Returns:
        patient_embeddings (np.ndarray): shape (n_patients, 512)
        channel_embeddings_all (np.ndarray): shape (n_patients, 43, 512)
    """
    print(f"\n📂 Processing: {file_path}")
    df = pd.read_csv(file_path)
    n_features = df.shape[1]
    data_array = df.to_numpy().reshape(n_patients, time_steps, n_features)

    patient_embeddings = []
    channel_embeddings_all = []

    for patient_ts in tqdm(data_array, desc="Encoding patients", leave=True, position=0):  # (49, 43)
        channel_embeddings = []

        for channel_idx in range(n_features):
            ts = patient_ts[:, channel_idx]  # shape: (49,)
            ts_tensor = torch.tensor(ts, dtype=torch.float32).unsqueeze(0)  # [1, 49]

            emb = pipeline.embed(context=ts_tensor)[0]  # shape: [1, 50, 512]
            emb_mean = emb.squeeze(0).mean(dim=0)       # shape: [512]

            emb_numpy = emb_mean.to(dtype=torch.float32).numpy()
            channel_embeddings.append(emb_numpy)

        channel_embeddings = np.stack(channel_embeddings)  # shape: (43, 512)
        channel_embeddings_all.append(channel_embeddings)

        patient_embedding = channel_embeddings.mean(axis=0)  # shape: (512,)
        patient_embeddings.append(patient_embedding)

    patient_embeddings = np.vstack(patient_embeddings)  # (4000, 512)
    channel_embeddings_all = np.stack(channel_embeddings_all)  # (4000, 43, 512)

    if save_prefix:
        np.save(f"{save_prefix}_patient_avg.npy", patient_embeddings)
        np.save(f"{save_prefix}_channelwise.npy", channel_embeddings_all)
        pd.DataFrame(patient_embeddings).to_csv(f"{save_prefix}_patient_avg.csv", index=False)
        print(f"Saved averaged embeddings to '{save_prefix}_patient_avg.npy' and CSV")
        print(f"Saved full channel-wise embeddings to '{save_prefix}_channelwise.npy'")

    return patient_embeddings, channel_embeddings_all

# Run for each dataset
X_train = extract_patient_embeddings("imputed_only_set-a.csv", pipeline, save_prefix="/cluster/scratch/gcardenal/chronos_embeddings_train")
X_val   = extract_patient_embeddings("imputed_only_set-b.csv", pipeline, save_prefix="/cluster/scratch/gcardenal/chronos_embeddings_val")
X_test  = extract_patient_embeddings("imputed_only_set-c.csv", pipeline, save_prefix="/cluster/scratch/gcardenal/chronos_embeddings_test")

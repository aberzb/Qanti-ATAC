import os
import random
import numpy as np
import torch
import yaml

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

@torch.no_grad()
def batch_metrics(y_true, y_pred):
    # y_true,y_pred: [B, bins]
    # simple calibration-friendly metrics
    mae = (y_true - y_pred).abs().mean().item()
    mse = ((y_true - y_pred) ** 2).mean().item()
    return {"mae": mae, "mse": mse}

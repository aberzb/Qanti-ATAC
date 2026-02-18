import argparse
import torch
from torch.utils.data import DataLoader

from src.data import QantiAtacDataset
from src.train import build_model
from src.utils import load_yaml, batch_metrics

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device(cfg.get("device", "cpu"))

    ds = QantiAtacDataset(args.npz, use_pwm=cfg["model"]["dna"]["use_pwm"])
    sample = ds[0]
    bins = sample["atac"].shape[-1]
    seq_len = sample["seq"].shape[-1]
    pwm_dim = sample["pwm"].shape[-1] if ("pwm" in sample) else None

    model = build_model(cfg, seq_len=seq_len, bins=bins, pwm_dim=pwm_dim).to(device)
    ck = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    loader = DataLoader(ds, batch_size=128, shuffle=False)

    all_pred = []
    all_true = []
    for batch in loader:
        atac = batch["atac"].to(device)
        seq = batch["seq"].to(device)
        y = batch["y"].to(device)
        pwm = batch.get("pwm")
        pwm = pwm.to(device) if pwm is not None else None

        pred = model(atac, seq, pwm=pwm)  # [0,1]
        if y.max() > 1.5:
            y = y / 100.0

        all_pred.append(pred.cpu())
        all_true.append(y.cpu())

    y_pred = torch.cat(all_pred, dim=0)
    y_true = torch.cat(all_true, dim=0)

    m = batch_metrics(y_true, y_pred)
    print("metrics:", m)
    print("example openness% first sample:", (y_pred[0] * 100.0).numpy())

if __name__ == "__main__":
    main()

import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import QantiAtacDataset
from src.models import MixedKernelATAC, DNAContextCNN, JointOpennessModel
from src.utils import set_seed, load_yaml, batch_metrics

def build_model(cfg, seq_len: int, bins: int, pwm_dim: int | None):
    atac_cfg = cfg["model"]["atac"]
    dna_cfg = cfg["model"]["dna"]
    joint_cfg = cfg["model"]["joint"]

    atac = MixedKernelATAC(
        in_channels=atac_cfg["in_channels"],
        channels=atac_cfg["channels"],
        kernels=atac_cfg["kernels"],
        dropout=atac_cfg["dropout"],
    )

    use_pwm = bool(dna_cfg["use_pwm"]) and (pwm_dim is not None)
    dna = DNAContextCNN(
        channels=dna_cfg["channels"],
        kernel_size=dna_cfg["kernel_size"],
        num_res_blocks=dna_cfg["num_res_blocks"],
        dropout=dna_cfg["dropout"],
        use_pwm=use_pwm,
        pwm_dim=(pwm_dim or dna_cfg["pwm_dim"]),
    )

    # dummy forward to get dims
    device = cfg.get("device", "cpu")
    atac_dummy = torch.zeros(2, 1, bins).to(device)
    seq_dummy = torch.zeros(2, 4, seq_len).to(device)
    pwm_dummy = torch.zeros(2, pwm_dim).to(device) if use_pwm else None

    atac.to(device); dna.to(device)
    with torch.no_grad():
        a = atac(atac_dummy)
        d = dna(seq_dummy, pwm=pwm_dummy)
        in_dim = a.shape[-1] + d.shape[-1]

    # build joint MLP with correct in_dim
    hidden = joint_cfg["hidden"]
    dropout = joint_cfg["dropout"]
    out_bins = joint_cfg["out_bins"]

    mlp = []
    prev = in_dim
    for h in hidden:
        mlp += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
        prev = h
    mlp += [nn.Linear(prev, out_bins)]

    model = JointOpennessModel(atac, dna, hidden=[], dropout=0.0, out_bins=out_bins)
    model.mlp = nn.Sequential(*mlp)
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--save", default="checkpoints/qanti_atac.pt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cpu"))

    ds_train = QantiAtacDataset(cfg["data"]["train_npz"], use_pwm=cfg["model"]["dna"]["use_pwm"])
    ds_val = QantiAtacDataset(cfg["data"]["val_npz"], use_pwm=cfg["model"]["dna"]["use_pwm"])

    # infer shapes
    sample = ds_train[0]
    bins = sample["atac"].shape[-1]
    seq_len = sample["seq"].shape[-1]
    pwm_dim = sample["pwm"].shape[-1] if ("pwm" in sample) else None

    model = build_model(cfg, seq_len=seq_len, bins=bins, pwm_dim=pwm_dim).to(device)

    train_loader = DataLoader(
        ds_train, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        ds_val, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True
    )

    loss_name = cfg["train"]["loss"]
    if loss_name == "bce":
        criterion = nn.BCELoss()
    elif loss_name == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = float("inf")
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch} train", leave=False)
        for step, batch in enumerate(pbar, start=1):
            atac = batch["atac"].to(device)         # [B,1,bins]
            seq = batch["seq"].to(device)           # [B,4,L]
            y = batch["y"].to(device)               # [B,bins]
            pwm = batch.get("pwm")
            pwm = pwm.to(device) if pwm is not None else None

            pred = model(atac, seq, pwm=pwm)        # [B,bins] sigmoid in [0,1]

            # If your y is stored as 0-100%, convert it here:
            if y.max() > 1.5:
                y = y / 100.0

            loss = criterion(pred, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % cfg["train"]["log_every"] == 0:
                m = batch_metrics(y.detach(), pred.detach())
                pbar.set_postfix(loss=float(loss.item()), **m)

        # validation
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch} val", leave=False):
                atac = batch["atac"].to(device)
                seq = batch["seq"].to(device)
                y = batch["y"].to(device)
                pwm = batch.get("pwm")
                pwm = pwm.to(device) if pwm is not None else None

                pred = model(atac, seq, pwm=pwm)
                if y.max() > 1.5:
                    y = y / 100.0
                loss = criterion(pred, y)

                val_loss += loss.item() * y.shape[0]
                n += y.shape[0]
        val_loss /= max(n, 1)
        print(f"epoch {epoch}: val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"cfg": cfg, "model": model.state_dict()}, args.save)
            print(f"  saved best -> {args.save}")

if __name__ == "__main__":
    main()

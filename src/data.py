import numpy as np
import torch
from torch.utils.data import Dataset

class QantiAtacDataset(Dataset):
    """
    Expects NPZ with keys:
      atac: [N, BINS]
      seq:  [N, L, 4] or [N, 4, L]
      pwm:  [N, D] (optional)
      y:    [N, BINS]
    """
    def __init__(self, npz_path: str, use_pwm: bool = True):
        super().__init__()
        z = np.load(npz_path, allow_pickle=False)

        self.atac = z["atac"].astype(np.float32)  # [N, bins]
        self.y = z["y"].astype(np.float32)

        seq = z["seq"]
        if seq.ndim != 3:
            raise ValueError(f"seq must be 3D [N,L,4] or [N,4,L], got {seq.shape}")
        # convert to [N,4,L]
        if seq.shape[-1] == 4:
            seq = np.transpose(seq, (0, 2, 1))
        self.seq = seq.astype(np.float32)

        self.use_pwm = use_pwm and ("pwm" in z.files)
        self.pwm = z["pwm"].astype(np.float32) if self.use_pwm else None

        if self.atac.shape[0] != self.seq.shape[0] or self.y.shape[0] != self.seq.shape[0]:
            raise ValueError("N mismatch among atac/seq/y")

        if self.use_pwm and self.pwm.shape[0] != self.seq.shape[0]:
            raise ValueError("N mismatch between pwm and seq")

        # basic sanity
        if self.seq.shape[1] != 4:
            raise ValueError(f"seq should have 4 channels, got {self.seq.shape}")

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx: int):
        atac = torch.from_numpy(self.atac[idx]).unsqueeze(0)  # [1, bins]
        seq = torch.from_numpy(self.seq[idx])                  # [4, L]
        y = torch.from_numpy(self.y[idx])                      # [bins]
        if self.use_pwm:
            pwm = torch.from_numpy(self.pwm[idx])              # [D]
            return {"atac": atac, "seq": seq, "pwm": pwm, "y": y}
        return {"atac": atac, "seq": seq, "y": y}

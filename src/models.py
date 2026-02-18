from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, dropout: float):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MixedKernelATAC(nn.Module):
    """
    ATAC signal module: multi-kernel conv over bins/neighborhood.
    Input:  x_atac [B, 1, bins]
    Output: features [B, F]
    """
    def __init__(self, in_channels: int, channels: List[int], kernels: List[int], dropout: float):
        super().__init__()
        if len(kernels) < 1:
            raise ValueError("kernels must be non-empty")

        # parallel convs at first layer (mixed kernels), then stack
        first_out = channels[0]
        self.branches = nn.ModuleList([
            ConvBlock1D(in_channels, first_out, k=k, dropout=dropout) for k in kernels
        ])
        merged_in = first_out * len(kernels)

        layers = []
        prev = merged_in
        for ch in channels[1:]:
            layers.append(ConvBlock1D(prev, ch, k=3, dropout=dropout))
            prev = ch
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B,1,bins]
        feats = [b(x) for b in self.branches]           # each [B, C, bins]
        x = torch.cat(feats, dim=1)                     # [B, C*#kernels, bins]
        x = self.backbone(x)                            # [B, C2, bins] or identity
        x = self.pool(x).squeeze(-1)                    # [B, C2]
        return x

class ResidualBlock(nn.Module):
    def __init__(self, ch: int, k: int, dropout: float):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        return F.relu(x + h, inplace=True)

class DNAContextCNN(nn.Module):
    """
    DNA context module:
      - sequence conv + residual blocks
      - optional PWM feature projection
    Input:
      seq: [B,4,L]
      pwm: [B,D] optional
    Output:
      features [B, F]
    """
    def __init__(self, channels: List[int], kernel_size: int, num_res_blocks: int,
                 dropout: float, use_pwm: bool, pwm_dim: int):
        super().__init__()
        self.use_pwm = use_pwm

        layers = []
        in_ch = 4
        for ch in channels:
            layers.append(nn.Conv1d(in_ch, ch, kernel_size=kernel_size, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            in_ch = ch
        self.stem = nn.Sequential(*layers)

        self.res = nn.Sequential(*[ResidualBlock(channels[-1], kernel_size, dropout) for _ in range(num_res_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)

        if self.use_pwm:
            self.pwm_proj = nn.Sequential(
                nn.Linear(pwm_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            self.out_dim = channels[-1] + 256
        else:
            self.pwm_proj = None
            self.out_dim = channels[-1]

    def forward(self, seq, pwm=None):
        x = self.stem(seq)                 # [B,C,L]
        x = self.res(x)                    # [B,C,L]
        x = self.pool(x).squeeze(-1)       # [B,C]
        if self.use_pwm:
            if pwm is None:
                raise ValueError("use_pwm=True but pwm=None")
            p = self.pwm_proj(pwm)         # [B,256]
            x = torch.cat([x, p], dim=-1)  # [B, C+256]
        return x

class JointOpennessModel(nn.Module):
    """
    Joint module:
      concat(ATAC_features, DNA_features) -> MLP -> sigmoid(out_bins)
    """
    def __init__(self, atac_module: MixedKernelATAC, dna_module: DNAContextCNN,
                 hidden: List[int], dropout: float, out_bins: int):
        super().__init__()
        self.atac = atac_module
        self.dna = dna_module

        in_dim = self._infer_in_dim()
        mlp = []
        prev = in_dim
        for h in hidden:
            mlp += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        mlp += [nn.Linear(prev, out_bins)]
        self.mlp = nn.Sequential(*mlp)

    def _infer_in_dim(self):
        # rough inference based on module definitions
        atac_dim = self.atac.pool.output_size * 0  # unused
        # we can’t know atac output channels directly without running a dummy forward,
        # so we’ll set at construction time via a dummy in train.py.
        return -1

    def forward(self, atac, seq, pwm=None):
        a = self.atac(atac)             # [B, Fa]
        d = self.dna(seq, pwm=pwm)      # [B, Fd]
        x = torch.cat([a, d], dim=-1)   # [B, Fa+Fd]
        logits = self.mlp(x)            # [B, out_bins]
        return torch.sigmoid(logits)    # [0,1]

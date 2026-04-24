import math

import torch
import torch.nn as nn


class MLPTimeConditioner(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, value):
        return self.net(value)


class SinusoidalTimeConditioner(nn.Module):
    def __init__(self, in_dim, out_dim, num_bands=8, max_frequency=128.0):
        super().__init__()
        if in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {in_dim}.")
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}.")
        if max_frequency <= 0:
            raise ValueError(f"max_frequency must be > 0, got {max_frequency}.")
        self.in_dim = int(in_dim)
        self.num_bands = int(num_bands)
        self.out_dim = int(out_dim)
        if self.num_bands == 1:
            frequencies = torch.ones(1, dtype=torch.float32)
        else:
            frequencies = torch.exp(torch.linspace(0.0, math.log(float(max_frequency)), self.num_bands))
        self.register_buffer("frequencies", frequencies)
        self.proj = nn.Linear(self.in_dim * self.num_bands * 2, self.out_dim)

    def forward(self, value):
        if value.shape[-1] != self.in_dim:
            raise ValueError(f"Expected last dimension {self.in_dim}, got {value.shape[-1]}.")
        frequencies = self.frequencies.to(device=value.device, dtype=value.dtype)
        angles = value.unsqueeze(-1) * frequencies * (2.0 * math.pi)
        features = torch.cat([angles.sin(), angles.cos()], dim=-1).reshape(*value.shape[:-1], -1)
        return self.proj(features)


def build_time_projector(in_dim, out_dim):
    return MLPTimeConditioner(in_dim=in_dim, out_dim=out_dim)


__all__ = [
    "MLPTimeConditioner",
    "SinusoidalTimeConditioner",
    "build_time_projector",
]

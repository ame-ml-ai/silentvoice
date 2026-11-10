"""Model-input transforms, numerically identical to Auto-AVSR ``VideoTransform``.

eval/test:  /255 -> CenterCrop(88) -> Grayscale -> Normalize(0.421, 0.165)
train:      /255 -> RandomCrop(88) -> Grayscale -> AdaptiveTimeMask(10, 25) -> Normalize

``AdaptiveTimeMask`` is re-implemented from Auto-AVSR (Apache-2.0, Imperial College
London) because importing upstream ``datamodule.transforms`` requires torchaudio.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T

MEAN, STD = 0.421, 0.165
INPUT_SIZE = 88


class AdaptiveTimeMask(torch.nn.Module):
    """Zero out random temporal spans; number of spans scales with length."""

    def __init__(self, window: int = 10, stride: int = 25) -> None:
        super().__init__()
        self.window, self.stride = window, stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (T, ...)
        cloned = x.clone()
        length = cloned.size(0)
        n_mask = int((length + self.stride - 0.1) // self.stride)
        for t, t_end in torch.randint(0, self.window, size=(n_mask, 2)):
            if length - t <= 0:
                continue
            t_start = random.randrange(0, length - int(t))
            if int(t) == 0:
                continue
            cloned[t_start : t_start + int(t_end)] = 0
        return cloned


class _Scale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / 255.0


def eval_transform() -> torch.nn.Module:
    return torch.nn.Sequential(_Scale(), T.CenterCrop(INPUT_SIZE), T.Grayscale(), T.Normalize(MEAN, STD))


def train_transform() -> torch.nn.Module:
    return torch.nn.Sequential(
        _Scale(), T.RandomCrop(INPUT_SIZE), T.Grayscale(), AdaptiveTimeMask(10, 25), T.Normalize(MEAN, STD)
    )


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T, 3, H, W) float tensor (0..255)."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected (T, H, W, 3) frames, got {frames.shape}.")
    return torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float()


def to_model_input(mouth_frames: np.ndarray) -> torch.Tensor:
    """Mouth crops (T, 96, 96, 3) uint8 -> model input (T, 1, 88, 88) float."""
    with torch.no_grad():
        return eval_transform()(frames_to_tensor(mouth_frames))

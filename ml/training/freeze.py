"""Partial fine-tuning strategies for the Auto-AVSR E2E model.

Module names are those of ``E2E`` in e2e_asr_conformer.py:
frontend (Conv3d + ResNet-18), proj_encoder, encoder.encoders[0..11], encoder.after_norm,
decoder (6 blocks + output layer), ctc.
"""

from __future__ import annotations

import torch

STRATEGIES = ("decoder_only", "encoder_top_k", "full")


def _set(module: torch.nn.Module, trainable: bool) -> None:
    for p in module.parameters():
        p.requires_grad = trainable


def apply_freeze(model: torch.nn.Module, strategy: str, encoder_trainable_blocks: int = 4) -> dict[str, object]:
    """Set ``requires_grad`` according to the strategy; returns per-component state."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown freeze strategy '{strategy}'. Choose from {STRATEGIES}.")
    _set(model, strategy == "full")
    if strategy == "full":
        return {"frontend": True, "proj_encoder": True, "encoder_blocks_trainable": len(model.encoder.encoders),
                "decoder": True, "ctc": True}

    _set(model.decoder, True)
    _set(model.ctc, True)
    state: dict[str, object] = {"frontend": False, "proj_encoder": False, "encoder_blocks_trainable": 0,
                                "decoder": True, "ctc": True}
    if strategy == "encoder_top_k":
        blocks = list(model.encoder.encoders)
        k = int(encoder_trainable_blocks)
        if not 0 < k <= len(blocks):
            raise ValueError(f"encoder_trainable_blocks must be in 1..{len(blocks)}")
        for block in blocks[-k:]:
            _set(block, True)
        if hasattr(model.encoder, "after_norm"):
            _set(model.encoder.after_norm, True)
        state["encoder_blocks_trainable"] = k
    return state


def freeze_batchnorm_stats(model: torch.nn.Module, all_layers: bool = True) -> int:
    """Put BatchNorm layers in eval mode (call after every ``model.train()``).

    BN layers whose parameters are frozen are *always* put in eval mode, because
    ``requires_grad=False`` does not stop running-statistics updates. ``all_layers``
    additionally freezes the statistics of trainable BN layers.
    """
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            frozen = not any(p.requires_grad for p in m.parameters())
            if all_layers or frozen:
                m.eval()
                n += 1
    return n

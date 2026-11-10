"""Owns the single loaded VSR model and reports its state honestly."""

from __future__ import annotations

import logging
import threading
from typing import Any

import torch

from backend.app.core.config import Settings
from ml.inference.model import CheckpointError, DecodeConfig, VSRModel, architecture_parameter_counts
from ml.utils.device import DeviceUnavailableError, resolve_device
from ml.utils.env import md5_file
from ml.utils.upstream import AUTO_AVSR_COMMIT

log = logging.getLogger("silentvoice.model")

# (stage name, description, key into count_parameters() or None)
ARCHITECTURE: list[tuple[str, str, str | None]] = [
    ("Input", "Grayscale mouth region, 88x88 px, 25 fps. Visual only: audio is never read.", None),
    ("Visual frontend", "Conv3d 1->64 (kernel 5x7x7, stride 1x2x2) + BN + SiLU + MaxPool3d, "
                        "then a ResNet-18 applied per frame with global average pooling -> 512-d", "frontend"),
    ("Projection", "Linear 512 -> 768", "proj_encoder"),
    ("Conformer encoder", "12 blocks, d=768, 12 heads, FFN 3072, depthwise conv kernel 31", "encoder"),
    ("CTC head", "Linear 768 -> 5049 tokens; weight 0.1 in loss and in decoding", "ctc"),
    ("Transformer decoder", "6 blocks, d=768, 12 heads, FFN 3072; weight 0.9", "decoder"),
    ("Beam search", "Joint CTC/attention, beam 40, no external language model", None),
    ("Output", "SentencePiece unigram tokens (5047 units) -> text", None),
]


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state = "not_configured"
        self.message = "No checkpoint configured. Set SILENTVOICE_CHECKPOINT_PATH to enable recognition."
        self.model: VSRModel | None = None
        self.checkpoint_md5: str | None = None
        self._lock = threading.Lock()
        if settings.checkpoint_path is not None and not settings.load_model_on_startup:
            self.message = f"Checkpoint {settings.checkpoint_path.name} configured but load_model_on_startup is false."
        try:
            self.device = resolve_device(settings.device)
        except DeviceUnavailableError as exc:
            self.device = torch.device("cpu")
            self.state, self.message = "error", str(exc)

    def load(self) -> None:
        ckpt = self.settings.checkpoint_path
        if ckpt is None or self.state == "error":
            return
        with self._lock:
            self.state, self.message = "loading", f"Loading {ckpt.name} on {self.device.type}..."
            try:
                self.model = VSRModel(ckpt, self.device, DecodeConfig(beam_size=self.settings.beam_size))
                self.checkpoint_md5 = md5_file(ckpt)
                self.state, self.message = "ready", f"{ckpt.name} loaded on {self.device.type}."
                log.info(self.message)
            except CheckpointError as exc:
                self.model = None
                self.state, self.message = "error", str(exc)
                log.error("Model load failed: %s", exc)
            except Exception as exc:  # OOM, CUDA errors, I/O: never leave the state stuck at "loading"
                self.model = None
                self.state, self.message = "error", f"Model load failed ({type(exc).__name__}); see the server log."
                log.exception("Model load failed")

    def info(self) -> dict[str, Any]:
        # architecture counts are a property of the code, valid with or without a checkpoint
        params = self.model.info()["parameters"] if self.model else architecture_parameter_counts()
        arch = [{"name": n, "detail": d, "parameters": params.get(k) if params and k else None}
                for n, d, k in ARCHITECTURE]
        base: dict[str, Any] = {
            "state": self.state, "message": self.message, "architecture": arch, "parameters": params,
            "upstream": {"repository": "https://github.com/mpc001/auto_avsr", "commit": AUTO_AVSR_COMMIT,
                         "license": "Apache-2.0 (code); checkpoint terms follow their training data"},
        }
        if self.model:
            info = self.model.info()
            base.update(checkpoint=info["checkpoint"], checkpoint_md5=self.checkpoint_md5, device=info["device"],
                        decode=info["decode"], parameters=info["parameters"], vocab_size=info["vocab_size"])
        return base

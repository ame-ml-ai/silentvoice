"""Auto-AVSR visual-only model: construction, checkpoint loading and decoding.

Verified architecture (third_party/auto_avsr/espnet/nets/pytorch_backend/e2e_asr_conformer.py):

* Frontend  : Conv3d(1->64, k=5x7x7, s=1x2x2) + BN + SiLU + MaxPool3d, then a 2D
              ResNet-18 trunk applied per frame, global average pool -> 512-d per frame
* Projection: Linear 512 -> 768
* Encoder   : Conformer, 12 blocks, d=768, 12 heads, FFN 3072, conv kernel 31
* Decoder   : Transformer, 6 blocks, d=768, 12 heads, FFN 3072
* Heads     : CTC (768 -> 5049) and attention decoder; loss 0.1*CTC + 0.9*attention
* Decoding  : joint CTC/attention beam search, beam 40, CTC weight 0.1, no LM
* Vocabulary: SentencePiece unigram, 5047 units + <blank> + <eos> = 5049
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ml.data.text import TextTokenizer, get_tokenizer
from ml.utils.device import describe_device
from ml.utils.upstream import ensure_auto_avsr_on_path


class CheckpointError(RuntimeError):
    """The checkpoint is missing, unreadable or incompatible with the architecture."""


@dataclass(frozen=True)
class DecodeConfig:
    beam_size: int = 40
    ctc_weight: float = 0.1
    length_penalty: float = 0.0


@dataclass(frozen=True)
class Transcription:
    text: str
    token_ids: list[int]
    num_frames: int
    inference_seconds: float


def build_model(vocab_size: int, ctc_weight: float = 0.1) -> torch.nn.Module:
    ensure_auto_avsr_on_path()
    from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E

    return E2E(vocab_size, "video", ctc_weight=ctc_weight)


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load weights from a model-zoo ``.pth`` or a SilentVoice training checkpoint.

    Uses ``weights_only=True`` so a checkpoint cannot execute code on load.
    """
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        obj: Any = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types here
        raise CheckpointError(f"Could not read checkpoint {path.name}: {exc}") from exc
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    elif isinstance(obj, dict) and "state_dict" in obj:  # Lightning checkpoint
        obj = {k.removeprefix("model."): v for k, v in obj["state_dict"].items()}
    if not isinstance(obj, dict) or not obj:
        raise CheckpointError(f"{path.name} does not contain a state dict.")
    return obj


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    def n(module: torch.nn.Module, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)

    counts = {name: n(getattr(model, name)) for name in ("frontend", "proj_encoder", "encoder", "decoder", "ctc")}
    counts["total"] = n(model)
    counts["trainable"] = n(model, trainable_only=True)
    return counts


_ARCH_COUNTS: dict[str, int] | None = None


def architecture_parameter_counts() -> dict[str, int]:
    """Exact parameter counts of the architecture, computed on the meta device (no weights allocated)."""
    global _ARCH_COUNTS
    if _ARCH_COUNTS is None:
        with torch.device("meta"):
            model = build_model(get_tokenizer().vocab_size)
        _ARCH_COUNTS = count_parameters(model)
    return dict(_ARCH_COUNTS)


def build_beam_search(model: torch.nn.Module, token_list: list[str], cfg: DecodeConfig) -> Any:
    """Same construction as Auto-AVSR ``lightning.get_beam_search_decoder``."""
    ensure_auto_avsr_on_path()
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus

    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(len(token_list))
    weights = {
        "decoder": 1.0 - cfg.ctc_weight,
        "ctc": cfg.ctc_weight,
        "lm": 0.0,
        "length_bonus": cfg.length_penalty,
    }
    return BatchBeamSearch(
        beam_size=cfg.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.odim - 1,
        eos=model.odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.ctc_weight == 1.0 else "decoder",
    )


class VSRModel:
    """A loaded visual speech recognition model ready for decoding."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        device: torch.device,
        decode: DecodeConfig = DecodeConfig(),
        tokenizer: TextTokenizer | None = None,
        allow_random_init: bool = False,
    ) -> None:
        self.tokenizer = tokenizer or get_tokenizer()
        self.device = device
        self.decode_config = decode
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model = build_model(self.tokenizer.vocab_size)
        if self.checkpoint_path is not None:
            state = load_state_dict(self.checkpoint_path)
            try:
                self.model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise CheckpointError(
                    f"{self.checkpoint_path.name} does not match the Auto-AVSR VSR architecture: {exc}"
                ) from exc
        elif not allow_random_init:
            raise CheckpointError("No checkpoint path given.")
        self.random_init = self.checkpoint_path is None
        self.model.to(device).eval()
        self._beam_search = build_beam_search(self.model, self.tokenizer.token_list, decode)
        self._beam_search.to(device)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """(T, 1, 88, 88) -> encoder features (T, 768)."""
        if video.ndim != 4 or video.shape[1:] != (1, 88, 88):
            raise ValueError(f"Expected model input (T, 1, 88, 88), got {tuple(video.shape)}.")
        x = self.model.frontend(video.unsqueeze(0).to(self.device))
        x = self.model.proj_encoder(x)
        enc, _ = self.model.encoder(x, None)
        return enc.squeeze(0)

    @torch.no_grad()
    def transcribe(self, video: torch.Tensor) -> Transcription:
        start = time.perf_counter()
        enc = self.encode(video)
        hyps = self._beam_search(enc)
        if not hyps:
            ids: list[int] = []
        else:
            ids = [int(i) for i in hyps[0].asdict()["yseq"][1:]]
        text = self.tokenizer.decode(torch.tensor(ids, dtype=torch.long))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return Transcription(text, ids, int(video.shape[0]), time.perf_counter() - start)

    def info(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint_path.name if self.checkpoint_path else None,
            "random_init": self.random_init,
            "device": describe_device(self.device),
            "decode": self.decode_config.__dict__,
            "parameters": count_parameters(self.model),
            "vocab_size": self.tokenizer.vocab_size,
        }

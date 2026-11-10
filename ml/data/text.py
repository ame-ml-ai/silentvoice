"""Text normalisation and the Auto-AVSR SentencePiece tokenizer.

Normalisation (used for labels and for scoring): upper-case, ``{...}`` annotation tags
removed (LRS3 ``{LG}``), typographic apostrophes mapped to ``'``, accents stripped, only
``[A-Z0-9' ]`` kept, whitespace collapsed.

The tokenizer reproduces upstream ``datamodule.transforms.TextTransform`` exactly
(unigram5000 model + units file from the pinned submodule):
``token_list = ["<blank>"] + units + ["<eos>"]`` -> 5049 entries.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import sentencepiece
import torch

from ml.utils.upstream import SPM_MODEL_PATH, SPM_UNITS_PATH, UpstreamMissingError

_TAG = re.compile(r"\{[^}]*\}")
_NOT_ALLOWED = re.compile(r"[^A-Z0-9' ]+")
_SPACES = re.compile(r"\s+")
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})


def normalize_text(text: str) -> str:
    text = _TAG.sub(" ", text).translate(_APOSTROPHES)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).upper()
    text = _NOT_ALLOWED.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


class TextTokenizer:
    """Text <-> token ids for the Auto-AVSR VSR checkpoints.

    Picklable (re-loads the SentencePiece model in the receiving process), so it can be
    held by a Dataset used with spawn-based DataLoader workers on Windows.
    """

    blank_id = 0
    ignore_id = -1

    def __init__(self, model_path: str | Path = SPM_MODEL_PATH, units_path: str | Path = SPM_UNITS_PATH) -> None:
        self.model_path, self.units_path = Path(model_path), Path(units_path)
        if not self.model_path.is_file() or not self.units_path.is_file():
            raise UpstreamMissingError(
                f"SentencePiece files not found ({self.model_path}). Run: git submodule update --init --recursive"
            )
        self._load()

    def _load(self) -> None:
        self.spm = sentencepiece.SentencePieceProcessor(model_file=str(self.model_path))
        units = self.units_path.read_text(encoding="utf8").splitlines()
        self.unit_to_id = {u.split()[0]: int(u.split()[-1]) for u in units if u.strip()}
        self.token_list = ["<blank>", *self.unit_to_id.keys(), "<eos>"]

    def __getstate__(self) -> dict[str, Path]:
        return {"model_path": self.model_path, "units_path": self.units_path}

    def __setstate__(self, state: dict[str, Path]) -> None:
        self.model_path, self.units_path = state["model_path"], state["units_path"]
        self._load()

    @property
    def vocab_size(self) -> int:
        return len(self.token_list)

    @property
    def eos_id(self) -> int:
        return self.vocab_size - 1

    def encode(self, text: str) -> torch.Tensor:
        unk = self.unit_to_id["<unk>"]
        pieces = self.spm.EncodeAsPieces(normalize_text(text))
        return torch.tensor([self.unit_to_id.get(p, unk) for p in pieces], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        ids = torch.as_tensor(ids, dtype=torch.long).flatten().tolist()
        specials = {self.ignore_id, self.blank_id, self.eos_id}
        text = "".join(self.token_list[i] for i in ids if i not in specials)
        return _SPACES.sub(" ", text.replace("<space>", " ").replace("▁", " ")).strip()


@lru_cache(maxsize=1)
def get_tokenizer() -> TextTokenizer:
    return TextTokenizer()

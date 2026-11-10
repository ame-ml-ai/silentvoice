"""Word and character error rate.

Corpus-level rates are computed as (S + D + I) / N summed over all utterances,
which is the convention used for LRS3 results (and by Auto-AVSR's ``eval.py``),
not the mean of per-utterance rates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import jiwer

from ml.data.text import normalize_text


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.reference_length == 0:
            raise ValueError("Reference is empty; error rate is undefined.")
        return self.errors / self.reference_length

    def to_dict(self) -> dict[str, float | int]:
        return {**asdict(self), "errors": self.errors, "rate": self.rate}


def _prepare(refs: Sequence[str], hyps: Sequence[str]) -> tuple[list[str], list[str]]:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references but {len(hyps)} hypotheses.")
    if not refs:
        raise ValueError("No utterances to score.")
    refs_n = [normalize_text(r) for r in refs]
    hyps_n = [normalize_text(h) for h in hyps]
    if any(not r for r in refs_n):
        raise ValueError("At least one reference is empty after normalisation.")
    return refs_n, hyps_n


def word_errors(refs: Sequence[str], hyps: Sequence[str]) -> ErrorCounts:
    refs_n, hyps_n = _prepare(refs, hyps)
    out = jiwer.process_words(refs_n, hyps_n)
    return ErrorCounts(
        out.substitutions, out.deletions, out.insertions, out.hits,
        out.substitutions + out.deletions + out.hits,
    )


def char_errors(refs: Sequence[str], hyps: Sequence[str]) -> ErrorCounts:
    """CER over characters, spaces included (jiwer's convention)."""
    refs_n, hyps_n = _prepare(refs, hyps)
    out = jiwer.process_characters(refs_n, hyps_n)
    return ErrorCounts(
        out.substitutions, out.deletions, out.insertions, out.hits,
        out.substitutions + out.deletions + out.hits,
    )


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return word_errors(refs, hyps).rate


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return char_errors(refs, hyps).rate

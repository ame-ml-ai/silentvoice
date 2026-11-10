"""JSONL manifests of preprocessed clips and split-leakage checks.

One row per clip::

    {"id": "grid/s3/bbaf2n", "dataset": "grid", "split": "val", "speaker": "s3",
     "video": "s3/bbaf2n.mp4", "num_frames": 75, "text": "BIN BLUE AT F TWO NOW"}

``video`` is relative to the dataset's ``videos/`` root (always with ``/`` separators).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from itertools import combinations
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")


class SplitLeakageError(ValueError):
    """The same speaker or clip id appears in more than one split (or twice)."""


@dataclass
class ManifestEntry:
    id: str
    dataset: str
    split: str
    speaker: str
    video: str
    num_frames: int
    text: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise ValueError(f"{self.id}: split must be one of {SPLITS}, got {self.split!r}.")
        for name in ("id", "dataset", "speaker", "video", "text"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{self.id!r}: field {name!r} must be a non-empty string.")
        if isinstance(self.num_frames, bool) or not isinstance(self.num_frames, int) or self.num_frames <= 0:
            raise ValueError(f"{self.id}: num_frames must be a positive integer, got {self.num_frames!r}.")


_FIELDS = tuple(f.name for f in fields(ManifestEntry))


def write_manifest(path: str | Path, entries: Iterable[ManifestEntry]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf8", newline="\n") as fh:
        for e in entries:
            fh.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run ml.preprocessing.prepare_dataset first.")
    out: list[ManifestEntry] = []
    with open(path, encoding="utf8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc}).") from exc
            if not isinstance(row, dict) or set(row) != set(_FIELDS):
                raise ValueError(f"{path}:{lineno}: expected exactly the fields {_FIELDS}.")
            try:
                out.append(ManifestEntry(**row))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return out


def select_split(entries: Iterable[ManifestEntry], split: str) -> list[ManifestEntry]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; use one of {SPLITS}.")
    return [e for e in entries if e.split == split]


@dataclass
class SplitReport:
    counts: dict[str, int]
    speakers: dict[str, int]
    speaker_overlap: dict[str, list[str]] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.speaker_overlap or self.duplicate_ids or self.missing_files)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "ok": self.ok}


def validate_splits(
    entries: Iterable[ManifestEntry],
    root: str | Path | None = None,
    require_speaker_disjoint: bool = True,
) -> SplitReport:
    """Count clips per split and detect speaker overlap, duplicate ids and missing files."""
    entries = list(entries)
    by_split: dict[str, set[str]] = {s: set() for s in SPLITS}
    seen: set[str] = set()
    dupes: list[str] = []
    for e in entries:
        by_split[e.split].add(e.speaker)
        if e.id in seen:
            dupes.append(e.id)
        seen.add(e.id)
    overlap: dict[str, list[str]] = {}
    if require_speaker_disjoint:
        for a, b in combinations(SPLITS, 2):
            common = sorted(by_split[a] & by_split[b])
            if common:
                overlap[f"{a}/{b}"] = common
    missing: list[str] = []
    if root is not None:
        root = Path(root)
        missing = [e.id for e in entries if not (root / e.video).is_file()]
    return SplitReport(
        counts={s: sum(e.split == s for e in entries) for s in SPLITS},
        speakers={s: len(v) for s, v in by_split.items()},
        speaker_overlap=overlap,
        duplicate_ids=sorted(set(dupes)),
        missing_files=missing,
    )


def assert_no_leakage(report: SplitReport) -> None:
    """Raise if splits share speakers or clip ids. Missing files are reported, not raised here."""
    problems = []
    if report.speaker_overlap:
        problems.append(f"speaker overlap {report.speaker_overlap}")
    if report.duplicate_ids:
        problems.append(f"duplicate ids {report.duplicate_ids[:5]}")
    if problems:
        raise SplitLeakageError("Split leakage: " + "; ".join(problems))

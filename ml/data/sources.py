"""Raw dataset discovery: GRID (primary) and LRS3 (optional).

GRID layout (Zenodo 3625687)::

    <raw>/<speaker>/*.mpg   and   <raw>/alignments/<speaker>/*.align

LRS3 layout::

    <raw>/{trainval,test}/<video_id>/<clip>.mp4 + <clip>.txt   (first line "Text:  ...")
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from ml.data.manifest import SPLITS
from ml.data.text import normalize_text

log = logging.getLogger(__name__)

GRID_SKIP_WORDS = {"sil", "sp"}

# GRID file names encode the sentence: command, colour, preposition, letter, digit, adverb
# (e.g. bbaf2n = "BIN BLUE AT F TWO NOW"). Verified on 2026-11-10: the decoded name equals the
# .align transcript for all 2,000 alignment files of s1 and s5.
_GRID_NAME = (
    {"b": "BIN", "l": "LAY", "p": "PLACE", "s": "SET"},
    {"b": "BLUE", "g": "GREEN", "r": "RED", "w": "WHITE"},
    {"a": "AT", "b": "BY", "i": "IN", "w": "WITH"},
    {c: c.upper() for c in "abcdefghijklmnopqrstuvxyz"},  # GRID letters exclude W
    {"z": "ZERO", "1": "ONE", "2": "TWO", "3": "THREE", "4": "FOUR", "5": "FIVE",
     "6": "SIX", "7": "SEVEN", "8": "EIGHT", "9": "NINE"},
    {"a": "AGAIN", "n": "NOW", "p": "PLEASE", "s": "SOON"},
)


@dataclass(frozen=True)
class RawClip:
    id: str
    dataset: str
    split: str
    speaker: str
    video_path: Path
    text: str
    rel_out: str  # output path relative to the processed videos/ root, "/" separators


def load_split_config(path: str | Path) -> dict[str, list[str]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf8")) or {}
    if set(raw) != set(SPLITS):
        raise ValueError(f"{path}: expected exactly the keys {SPLITS}, got {sorted(raw)}.")
    splits = {s: [str(x) for x in (raw[s] or [])] for s in SPLITS}
    seen: dict[str, str] = {}
    for split, speakers in splits.items():
        for spk in speakers:
            if spk in seen:
                raise ValueError(f"{path}: speaker {spk} is in both {seen[spk]} and {split}.")
            seen[spk] = split
    return splits


def parse_grid_alignment(path: str | Path) -> str:
    words = []
    for line in Path(path).read_text(encoding="utf8").splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[2].lower() not in GRID_SKIP_WORDS:
            words.append(parts[2])
    return normalize_text(" ".join(words))


def parse_grid_filename(stem: str) -> str:
    """Transcript encoded in a GRID clip name, e.g. ``bbaf2n`` -> ``BIN BLUE AT F TWO NOW``."""
    stem = stem.lower()
    if len(stem) != 6:
        raise ValueError(f"Not a GRID clip name: {stem!r}")
    try:
        return " ".join(table[ch] for table, ch in zip(_GRID_NAME, stem))
    except KeyError as exc:
        raise ValueError(f"Not a GRID clip name: {stem!r}") from exc


def parse_lrs3_text(path: str | Path) -> str:
    first = Path(path).read_text(encoding="utf8").splitlines()[0]
    if not first.startswith("Text:"):
        raise ValueError(f"{path}: first line does not start with 'Text:'.")
    return normalize_text(first[len("Text:"):])


def iter_grid(raw_dir: str | Path, split_config: str | Path) -> Iterator[RawClip]:
    raw_dir = Path(raw_dir)
    splits = load_split_config(split_config)
    for split in SPLITS:
        for spk in splits[split]:
            spk_dir = raw_dir / spk
            videos = sorted(spk_dir.rglob("*.mpg")) if spk_dir.is_dir() else []
            if not videos:
                log.warning("GRID %s: no .mpg files under %s", spk, spk_dir)
                continue
            from_name = skipped = mismatched = 0
            for video in videos:
                align = raw_dir / "alignments" / spk / f"{video.stem}.align"
                try:
                    name_text = parse_grid_filename(video.stem)
                except ValueError:
                    name_text = None
                text = parse_grid_alignment(align) if align.is_file() else ""
                if text and name_text and text != name_text:
                    mismatched += 1  # alignment wins; counted so a corrupt release is noticed
                if not text:
                    # Zenodo's alignments/s5 holds a different set of sentences than s5.zip
                    # (0 of 1,000 names match), so fall back to the sentence in the file name.
                    if name_text is None:
                        skipped += 1
                        continue
                    text, from_name = name_text, from_name + 1
                yield RawClip(f"grid/{spk}/{video.stem}", "grid", split, spk, video, text, f"{spk}/{video.stem}.mp4")
            if from_name or skipped or mismatched:
                log.warning("GRID %s: %d transcripts from file names (no .align), %d skipped, "
                            "%d alignment/name mismatches", spk, from_name, skipped, mismatched)


def _lrs3_is_val(video_id: str, fraction: float) -> bool:
    h = int(hashlib.md5(video_id.encode("utf8")).hexdigest()[:8], 16)  # noqa: S324 - deterministic split only
    return h / 0xFFFFFFFF < fraction


def iter_lrs3(raw_dir: str | Path, val_fraction: float = 0.02) -> Iterator[RawClip]:
    """trainval -> train/val (deterministic ~val_fraction of video ids), test -> test. pretrain is unused."""
    raw_dir = Path(raw_dir)
    for part in ("trainval", "test"):
        for video in sorted((raw_dir / part).glob("*/*.mp4")):
            txt = video.with_suffix(".txt")
            if not txt.is_file():
                continue
            vid = video.parent.name
            split = "test" if part == "test" else ("val" if _lrs3_is_val(vid, val_fraction) else "train")
            yield RawClip(f"lrs3/{part}/{vid}/{video.stem}", "lrs3", split, vid, video,
                          parse_lrs3_text(txt), f"{part}/{vid}/{video.stem}.mp4")

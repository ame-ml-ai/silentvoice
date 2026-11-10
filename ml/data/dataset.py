"""Training dataset, padding collate and a frame-budget batch sampler."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from ml.data.manifest import ManifestEntry
from ml.data.text import TextTokenizer
from ml.preprocessing.transforms import frames_to_tensor
from ml.preprocessing.video_io import read_video_rgb


class VSRDataset(Dataset):
    """Preprocessed mouth-crop clips -> {"input": (T, 1, 88, 88), "target": (L,), "id", "text"}."""

    def __init__(self, entries: Sequence[ManifestEntry], data_root: str | Path,
                 tokenizer: TextTokenizer, transform: torch.nn.Module) -> None:
        self.entries = list(entries)
        self.data_root = Path(data_root)
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, object]:
        e = self.entries[idx]
        frames, _ = read_video_rgb(self.data_root / e.video)
        with torch.no_grad():
            video = self.transform(frames_to_tensor(frames))
        return {"input": video, "target": self.tokenizer.encode(e.text), "id": e.id, "text": e.text}


def collate_pad(batch: list[dict[str, object]]) -> dict[str, object]:
    """Zero-pad inputs and -1-pad targets (Auto-AVSR convention)."""
    inputs = [b["input"] for b in batch]
    targets = [b["target"] for b in batch]
    t_max = max(int(x.shape[0]) for x in inputs)
    x = inputs[0].new_zeros((len(inputs), t_max, *inputs[0].shape[1:]))
    for i, v in enumerate(inputs):
        x[i, : v.shape[0]] = v
    l_max = max(int(t.shape[0]) for t in targets)
    y = torch.full((len(targets), l_max), -1, dtype=torch.long)
    for i, t in enumerate(targets):
        y[i, : t.shape[0]] = t
    return {
        "inputs": x,
        "input_lengths": torch.tensor([int(v.shape[0]) for v in inputs], dtype=torch.long),
        "targets": y,
        "target_lengths": torch.tensor([int(t.shape[0]) for t in targets], dtype=torch.long),
        "ids": [b["id"] for b in batch],
        "texts": [b["text"] for b in batch],
    }


class FrameBudgetBatchSampler(Sampler[list[int]]):
    """Batches whose padded size (max length x batch size) stays within ``max_frames``.

    Each epoch: indices are shuffled (seed + epoch), sorted by length inside windows of
    ``window`` clips to limit padding, packed greedily, and the batch order is shuffled.
    Shuffling first matters for GRID: the manifest is speaker-contiguous and all clips
    have the same length, so without it every batch would hold a single speaker.
    """

    def __init__(self, lengths: Sequence[int], max_frames: int, shuffle: bool = True, seed: int = 0,
                 max_batch_size: int | None = None, window: int = 1000) -> None:
        self.lengths = [int(n) for n in lengths]
        too_long = [i for i, n in enumerate(self.lengths) if n > max_frames]
        if too_long:
            raise ValueError(f"{len(too_long)} clips exceed max_frames={max_frames} "
                             f"(longest {max(self.lengths)}); raise max_frames or drop them.")
        self.max_frames, self.shuffle, self.seed = int(max_frames), shuffle, seed
        self.max_batch_size, self.window = max_batch_size, window
        self.epoch = 0
        self._cache: tuple[int, list[list[int]]] | None = None

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        if self._cache is not None and self._cache[0] == self.epoch:
            return self._cache[1]
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        sorted_order: list[int] = []
        for start in range(0, len(order), self.window):
            chunk = order[start : start + self.window]
            sorted_order.extend(sorted(chunk, key=lambda i: self.lengths[i]))  # stable: keeps shuffle among ties
        batches: list[list[int]] = []
        cur: list[int] = []
        cur_max = 0
        for i in sorted_order:
            new_max = max(cur_max, self.lengths[i])
            full = self.max_batch_size is not None and len(cur) >= self.max_batch_size
            if cur and (new_max * (len(cur) + 1) > self.max_frames or full):
                batches.append(cur)
                cur, new_max = [], self.lengths[i]
            cur.append(i)
            cur_max = new_max
        if cur:
            batches.append(cur)
        if self.shuffle:
            rng.shuffle(batches)
        self._cache = (self.epoch, batches)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        return iter([list(b) for b in self._batches()])

    def __len__(self) -> int:
        return len(self._batches())

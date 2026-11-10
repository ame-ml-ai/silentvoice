"""Reads experiment outputs written by ml/evaluation/evaluate.py and ml/training/finetune.py.

Only ``results.json`` files with ``status == "complete"`` count as results. Smoke runs
are listed but flagged, and nothing is ever synthesised: a missing file is "pending".
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("silentvoice.experiments")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        log.warning("Ignoring malformed %s", path)
        return None


def _record(exp_id: str, kind: str, path: Path, data: dict[str, Any] | None, root: Path,
            training: dict[str, Any] | None = None) -> dict[str, Any]:
    rel = path.relative_to(root.parent).as_posix() if path.is_relative_to(root.parent) else path.as_posix()
    if data is None:
        return {"id": exp_id, "kind": kind, "status": "pending", "path": rel, "training": training}
    params = data.get("parameters", {})
    return {
        "id": exp_id, "kind": kind, "status": data.get("status", "pending"), "path": rel,
        "split": data.get("split"), "num_samples": data.get("num_samples"),
        "wer": data.get("wer"), "cer": data.get("cer"),
        "latency_mean_s": data.get("latency", {}).get("mean_s"),
        "real_time_factor": data.get("latency", {}).get("real_time_factor"),
        "peak_gpu_gb": data.get("memory", {}).get("peak_gpu_allocated_gb"),
        "parameters_total": params.get("total"),
        "parameters_trainable": (training or {}).get("parameters", {}).get("trainable", params.get("trainable")),
        "checkpoint": data.get("checkpoint", {}).get("name"),
        "device": data.get("device", {}).get("name"),
        "training": training,
    }


def _training_summary(directory: Path) -> dict[str, Any] | None:
    s = _read_json(directory / "training_summary.json")
    if not s or s.get("status") == "smoke":
        return None
    keep = ("status", "best_val_wer", "epochs_run", "optimizer_steps", "training_time_s",
            "peak_gpu_allocated_gb", "parameters", "freeze", "train_clips", "amp")
    return {k: s.get(k) for k in keep}


class ExperimentService:
    def __init__(self, root: Path) -> None:
        self.root = root

    def list(self) -> list[dict[str, Any]]:
        out = []
        for exp_id, kind in (("baseline", "baseline"), ("finetuning", "finetuning")):
            d = self.root / exp_id
            training = _training_summary(d) if kind == "finetuning" else None
            out.append(_record(exp_id, kind, d / "results.json", _read_json(d / "results.json"), self.root, training))
            smoke = _read_json(d / "results_smoke.json")
            if smoke:
                out.append(_record(f"{exp_id}_smoke", kind, d / "results_smoke.json", smoke, self.root))
        abl = self.root / "ablations"
        if abl.is_dir():
            for d in sorted(p for p in abl.iterdir() if p.is_dir()):
                out.append(_record(f"ablation/{d.name}", "ablation", d / "results.json",
                                   _read_json(d / "results.json"), self.root, _training_summary(d)))
        return out

    def _curves(self) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        path = self.root / "finetuning" / "train_log.jsonl"
        loss, val = [], []
        if not path.is_file():
            return loss, val
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("type") == "val":
                val.append({"epoch": row["epoch"], "val_loss": row["val_loss"], "val_wer": row["val_wer"]})
            elif "loss" in row:
                loss.append({"step": row["step"], "loss": row["loss"], "lr": row["lr"]})
        return loss, val

    def compare(self) -> dict[str, Any]:
        recs = {r["id"]: r for r in self.list()}
        base, ft = recs["baseline"], recs["finetuning"]
        reasons = []
        if base["status"] != "complete":
            reasons.append("Baseline evaluation has not been run on the RTX 5070 yet.")
        if ft["status"] != "complete":
            reasons.append("Fine-tuned model has not been evaluated yet.")
        if not reasons:
            if base["split"] != ft["split"] or base["num_samples"] != ft["num_samples"]:
                reasons.append("Baseline and fine-tuned results were not computed on the same split/sample count.")
        loss_curve, val_curve = self._curves()
        if reasons:
            return {"status": "pending", "reasons": reasons, "baseline": base, "finetuned": ft,
                    "loss_curve": loss_curve, "val_curve": val_curve}

        def rel(a: float, b: float) -> float | None:
            return (b - a) / a if a else None

        return {
            "status": "complete", "reasons": [], "baseline": base, "finetuned": ft,
            "wer_absolute_change": ft["wer"] - base["wer"], "wer_relative_change": rel(base["wer"], ft["wer"]),
            "cer_absolute_change": ft["cer"] - base["cer"], "cer_relative_change": rel(base["cer"], ft["cer"]),
            "loss_curve": loss_curve, "val_curve": val_curve,
        }

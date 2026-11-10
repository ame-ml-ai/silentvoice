"""Fine-tune a pretrained Auto-AVSR VSR checkpoint.

    # RTX 5070
    python -m ml.training.finetune --config ml/configs/finetune_grid.yaml

    # Mac: verify the loop end-to-end on CPU (2 optimizer steps, 8 clips). Results are
    # written under <output_dir>/smoke/ and are not experiment results.
    python -m ml.training.finetune --config ml/configs/finetune_grid.yaml --cpu-smoke

Without ``--cpu-smoke`` the script requires CUDA and exits if it is missing; it never
falls back to CPU training silently.

Plain PyTorch loop (no Lightning/DDP/wandb): single GPU, AMP, gradient accumulation,
partial freezing, warmup + cosine LR, gradient clipping, validation loss and validation
WER each epoch, best checkpoint chosen on validation WER, early stopping, resumable.
The test split is never loaded here.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from ml.data.dataset import FrameBudgetBatchSampler, VSRDataset, collate_pad
from ml.data.manifest import assert_no_leakage, read_manifest, select_split, validate_splits
from ml.data.text import get_tokenizer
from ml.evaluation.metrics import word_errors
from ml.inference.model import DecodeConfig, build_beam_search, build_model, count_parameters, load_state_dict
from ml.preprocessing.transforms import eval_transform, train_transform
from ml.training.freeze import apply_freeze, freeze_batchnorm_stats
from ml.utils.device import DeviceUnavailableError, describe_device, resolve_device
from ml.utils.env import environment_record, md5_file, set_seed

log = logging.getLogger("finetune")


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", encoding="utf8")

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row) + "\n")
        self.handle.flush()


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return f


def autocast_ctx(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def gpu_mem_gb(device: torch.device) -> float | None:
    return round(torch.cuda.max_memory_allocated() / 1024**3, 3) if device.type == "cuda" else None


@torch.no_grad()
def validate(model, loader, device, amp, beam_search, tokenizer, max_samples: int) -> dict[str, float]:
    model.eval()
    losses, refs, hyps, seen = [], [], [], 0
    for batch in loader:
        x, lens, y = batch["inputs"].to(device), batch["input_lengths"].to(device), batch["targets"].to(device)
        with autocast_ctx(device, amp):
            loss, _, _, _ = model(x, lens, y)
        losses.append(float(loss))
        for i in range(x.shape[0]):
            if seen >= max_samples:
                break
            feats = model.encoder(model.proj_encoder(model.frontend(x[i : i + 1, : lens[i]])), None)[0].squeeze(0)
            hyp = beam_search(feats.float())
            ids = torch.tensor([int(t) for t in hyp[0].asdict()["yseq"][1:]]) if hyp else torch.tensor([], dtype=torch.long)
            refs.append(batch["texts"][i])
            hyps.append(tokenizer.decode(ids))
            seen += 1
    return {"val_loss": sum(losses) / max(1, len(losses)), "val_wer": word_errors(refs, hyps).rate, "val_wer_samples": len(refs)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--cpu-smoke", action="store_true", help="explicit CPU smoke test: 2 steps on 8 clips")
    p.add_argument("--resume", type=Path, help="path to last.pth to resume from")
    p.add_argument("--max-steps", type=int, help="stop after N optimizer steps (memory/speed probes)")
    p.add_argument("--random-init", action="store_true",
                   help="with --cpu-smoke only: skip loading the checkpoint (loop test before it is downloaded)")
    args = p.parse_args(argv)
    if args.random_init and not args.cpu_smoke:
        p.error("--random-init is only allowed together with --cpu-smoke")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(args.config.read_text())
    tcfg, ocfg, fcfg = cfg["train"], cfg["optim"], cfg["freeze"]
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    if args.cpu_smoke:
        device = torch.device("cpu")
        out_dir = Path(cfg["output_dir"]) / "smoke"
        max_steps = args.max_steps or 2
        tcfg = {**tcfg, "grad_accum_steps": 1, "max_batch_size": 2, "val_max_samples": 2, "val_beam_size": 2}
        log.warning("CPU SMOKE TEST: verifies the training loop only; no result from this run is an experiment result.")
    else:
        try:
            device = resolve_device(cfg.get("device", "cuda"))
        except DeviceUnavailableError as exc:
            log.error("%s  (Use --cpu-smoke to test the loop on CPU.)", exc)
            return 3
        if device.type != "cuda":
            log.error("Fine-tuning requires CUDA (config device=%s). Use --cpu-smoke for a loop test.", device)
            return 3
        out_dir = Path(cfg["output_dir"])
        max_steps = args.max_steps
    out_dir.mkdir(parents=True, exist_ok=True)
    amp = "none" if device.type != "cuda" else tcfg.get("amp_dtype", "bf16")

    # ---- data: train + val only; leakage checks before anything is loaded
    entries = read_manifest(cfg["manifest"])
    train_val = [e for e in entries if e.split in ("train", "val")]
    report = validate_splits(entries)
    assert_no_leakage(report)
    missing = validate_splits(train_val, root=cfg["data_root"], require_speaker_disjoint=False).missing_files
    if missing:
        log.error("%d train/val clips listed in the manifest are missing on disk, e.g. %s", len(missing), missing[:3])
        return 4
    train_entries, val_entries = select_split(entries, "train"), select_split(entries, "val")
    if args.cpu_smoke:
        train_entries, val_entries = train_entries[:8], val_entries[:4]
    rng = random.Random(seed)
    val_wer_entries = val_entries if len(val_entries) <= tcfg["val_max_samples"] else rng.sample(val_entries, tcfg["val_max_samples"])

    tokenizer = get_tokenizer()
    train_ds = VSRDataset(train_entries, cfg["data_root"], tokenizer, train_transform())
    val_ds = VSRDataset(val_wer_entries, cfg["data_root"], tokenizer, eval_transform())
    sampler = FrameBudgetBatchSampler([e.num_frames for e in train_entries], tcfg["max_frames_per_batch"],
                                      shuffle=True, seed=seed, max_batch_size=tcfg.get("max_batch_size"))
    workers = 0 if args.cpu_smoke else tcfg.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_pad, num_workers=workers,
                              pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_pad, num_workers=workers)

    # ---- model
    model = build_model(tokenizer.vocab_size)
    if args.random_init:
        log.warning("Random initialisation (smoke test): no pretrained weights loaded.")
    else:
        model.load_state_dict(load_state_dict(cfg["checkpoint"]), strict=True)
    freeze_state = apply_freeze(model, fcfg["strategy"], fcfg.get("encoder_trainable_blocks", 4))
    model.to(device)
    params = count_parameters(model)
    log.info("Parameters: %s", params)

    trainable = [p_ for p_ in model.parameters() if p_.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=ocfg["lr"], weight_decay=ocfg["weight_decay"], betas=tuple(ocfg["betas"]))
    accum = int(tcfg["grad_accum_steps"])
    steps_per_epoch = max(1, len(sampler) // accum)  # the trailing len % accum micro-batches are dropped
    total_steps = max_steps or steps_per_epoch * int(tcfg["max_epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(int(ocfg["warmup_steps"]), total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == "fp16")
    beam = build_beam_search(model, tokenizer.token_list, DecodeConfig(beam_size=tcfg["val_beam_size"]))
    beam.to(device)

    start_epoch, step, best_wer, patience, skipped_steps = 0, 0, float("inf"), 0, 0
    resume_keys = {"freeze": fcfg, "optim": ocfg, "grad_accum_steps": tcfg["grad_accum_steps"],
                   "max_epochs": tcfg["max_epochs"], "checkpoint": cfg["checkpoint"]}
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        saved = json.loads(state["resume_keys"])
        if saved != json.loads(json.dumps(resume_keys)):
            log.error("Config differs from the run being resumed; refusing to resume.\n saved=%s\n now=%s", saved, resume_keys)
            return 5
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        start_epoch, step, best_wer, patience = state["epoch"] + 1, state["step"], state["best_val_wer"], state["patience"]
        log.info("Resumed from %s at epoch %d step %d", args.resume, start_epoch, step)

    train_log = JsonlLogger(out_dir / "train_log.jsonl")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    history: list[dict[str, Any]] = []
    stop = False
    for epoch in range(start_epoch, int(tcfg["max_epochs"])):
        sampler.set_epoch(epoch)
        model.train()
        freeze_batchnorm_stats(model, all_layers=fcfg.get("freeze_batchnorm", True))
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            x, lens, y = batch["inputs"].to(device), batch["input_lengths"].to(device), batch["targets"].to(device)
            with autocast_ctx(device, amp):
                loss, loss_ctc, loss_att, acc = model(x, lens, y)
            scaler.scale(loss / accum).backward()
            if (micro + 1) % accum != 0:
                continue
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, ocfg["grad_clip"])
            if not torch.isfinite(grad_norm) and amp != "fp16":
                # bf16/fp32 have no GradScaler to skip the step; never write NaN/Inf into the weights
                skipped_steps += 1
                log.warning("Non-finite gradient norm at step %d; update skipped (%d so far).", step, skipped_steps)
                optimizer.zero_grad(set_to_none=True)
                if skipped_steps > 50:
                    log.error("Too many non-finite gradients; aborting. Lower the LR or use amp_dtype: none.")
                    return 6
                continue
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % tcfg.get("log_every_steps", 20) == 0 or args.cpu_smoke:
                row = {"step": step, "epoch": epoch, "loss": float(loss), "loss_ctc": float(loss_ctc),
                       "loss_att": float(loss_att), "decoder_acc": float(acc), "grad_norm": float(grad_norm),
                       "lr": scheduler.get_last_lr()[0], "gpu_mem_gb": gpu_mem_gb(device),
                       "elapsed_s": round(time.time() - t_start, 1)}
                train_log.write(row)
                log.info("step %d loss %.4f lr %.2e mem %s", step, row["loss"], row["lr"], row["gpu_mem_gb"])
            if max_steps and step >= max_steps:
                stop = True
                break

        metrics = validate(model, val_loader, device, amp, beam, tokenizer, tcfg["val_max_samples"])
        metrics.update(epoch=epoch, step=step, elapsed_s=round(time.time() - t_start, 1), gpu_mem_gb=gpu_mem_gb(device))
        history.append(metrics)
        train_log.write({"type": "val", **metrics})
        log.info("epoch %d val_loss %.4f val_wer %.4f", epoch, metrics["val_loss"], metrics["val_wer"])

        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        improved = metrics["val_wer"] < best_wer
        if improved:
            best_wer, patience = metrics["val_wer"], 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "step": step,
                        "val_wer": best_wer, "base_checkpoint": Path(cfg["checkpoint"]).name}, ckpt_dir / "best.pth")
        else:
            patience += 1
        if not stop:  # only full epochs are resumable; a --max-steps stop is a probe, not a checkpoint
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(), "scaler_state_dict": scaler.state_dict(),
                        "epoch": epoch, "step": step, "best_val_wer": best_wer, "patience": patience,
                        "resume_keys": json.dumps(resume_keys)}, ckpt_dir / "last.pth")
        if stop or patience >= tcfg.get("early_stopping_patience", 3):
            break

    summary = {
        "status": "smoke" if args.cpu_smoke else ("partial" if max_steps else "complete"),
        "experiment": cfg.get("experiment"),
        "best_val_wer": best_wer,
        "epochs_run": len(history),
        "optimizer_steps": step,
        "skipped_nonfinite_steps": skipped_steps,
        "val_speakers": sorted({e.speaker for e in val_wer_entries}),
        "training_time_s": round(time.time() - t_start, 1),
        "peak_gpu_allocated_gb": gpu_mem_gb(device),
        "parameters": params,
        "freeze": freeze_state,
        "train_clips": len(train_entries),
        "val_clips_for_wer": len(val_wer_entries),
        "base_checkpoint": None if args.random_init else {"name": Path(cfg["checkpoint"]).name, "md5": md5_file(cfg["checkpoint"])},
        "device": describe_device(device),
        "amp": amp,
        "history": history,
        "config": cfg,
        "environment": environment_record(),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Finished: best val WER %.4f; summary in %s", best_wer, out_dir / "training_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

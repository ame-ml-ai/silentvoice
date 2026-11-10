"""Measure peak VRAM of one training step at several frame budgets (CUDA only).

    python scripts/gpu_memory_probe.py --strategy encoder_top_k --blocks 4 --amp bf16 \
        --frames 150 300 600 900 1200

Runs forward + backward + AdamW step on random input shaped like real batches
(clips of --clip-frames frames) with the real 250M-parameter model and the chosen
freezing strategy. Writes experiments/hardware/memory_probe.json. Use the largest
budget whose peak stays below --budget-gb (default: 80% of VRAM, ~6.5 GB on an 8 GB
RTX 4060 Laptop) as max_frames_per_batch in the fine-tune config; the script prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from ml.data.text import get_tokenizer  # noqa: E402
from ml.inference.model import build_model, count_parameters  # noqa: E402
from ml.training.freeze import apply_freeze  # noqa: E402
from ml.utils.device import describe_device, resolve_device  # noqa: E402


def probe(model, frames_per_batch: int, clip_frames: int, amp: str, vocab: int) -> dict[str, object]:
    device = torch.device("cuda")
    bsz = max(1, frames_per_batch // clip_frames)
    x = torch.randn(bsz, clip_frames, 1, 88, 88, device=device)
    lens = torch.full((bsz,), clip_frames, device=device)
    y = torch.randint(1, vocab - 1, (bsz, 12), device=device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        ctx = torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16) if amp != "none" else torch.enable_grad()
        with ctx:
            loss = model(x, lens, y)[0]
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return {"frames_per_batch": bsz * clip_frames, "batch_size": bsz, "ok": True,
                "peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 3)}
    except torch.OutOfMemoryError:
        return {"frames_per_batch": bsz * clip_frames, "batch_size": bsz, "ok": False, "peak_gb": None, "error": "OOM"}
    finally:
        del x, y, opt
        torch.cuda.empty_cache()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--strategy", default="encoder_top_k", choices=["decoder_only", "encoder_top_k", "full"])
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    p.add_argument("--clip-frames", type=int, default=75, help="75 = one GRID clip (3 s)")
    p.add_argument("--frames", type=int, nargs="+", default=[150, 300, 600, 900, 1200])
    p.add_argument("--budget-gb", type=float, help="peak-VRAM budget for the recommendation (default 80%% of VRAM)")
    p.add_argument("--out", type=Path, default=Path("experiments/hardware/memory_probe.json"))
    args = p.parse_args()

    device = resolve_device("cuda")  # raises if CUDA is missing
    vocab = get_tokenizer().vocab_size
    model = build_model(vocab).to(device).train()
    freeze = apply_freeze(model, args.strategy, args.blocks)
    rows = []
    for f in args.frames:
        row = probe(model, f, args.clip_frames, args.amp, vocab)
        print(row, flush=True)
        rows.append(row)
        if not row["ok"]:
            break
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    budget = args.budget_gb or round(0.8 * total_gb, 2)
    fitting = [r for r in rows if r["ok"] and r["peak_gb"] <= budget]
    best = max(fitting, key=lambda r: r["frames_per_batch"]) if fitting else None
    recommendation = None if best is None else {
        "budget_gb": budget, "max_frames_per_batch": best["frames_per_batch"], "max_batch_size": best["batch_size"],
        "grad_accum_steps_for_32_clips": max(1, round(32 / best["batch_size"])),
    }
    print(f"Recommendation (peak <= {budget} GB): {recommendation}" if recommendation
          else f"No tested budget fits in {budget} GB: use --strategy decoder_only or smaller --frames.")
    out = {"device": describe_device(device), "strategy": args.strategy, "recommendation": recommendation, "freeze": freeze, "amp": args.amp,
           "parameters": count_parameters(model), "clip_frames": args.clip_frames, "rows": rows,
           "note": "random inputs; real batches add data-loader memory on the host, not the GPU"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

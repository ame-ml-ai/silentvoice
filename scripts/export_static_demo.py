"""Export the API responses the frontend needs into static JSON for a backend-less deploy (Vercel).

    python scripts/export_static_demo.py                         # pilot results (default)
    python scripts/export_static_demo.py --experiments experiments

The real FastAPI app is run in-process (TestClient) with no checkpoint, so the files are
exactly what the API returns: nothing is hand-written. Output: frontend/public/demo/*.json,
read by the frontend when it is built with VITE_STATIC_DEMO=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ENDPOINTS = {
    "health": "/api/health",
    "model-info": "/api/model/info",
    "experiments": "/api/experiments",
    "compare": "/api/experiments/compare",
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiments", type=Path, default=ROOT / "experiments" / "pilot")
    p.add_argument("--out", type=Path, default=ROOT / "frontend" / "public" / "demo")
    args = p.parse_args()

    os.environ["SILENTVOICE_EXPERIMENTS_DIR"] = str(args.experiments.resolve())
    os.environ["SILENTVOICE_CHECKPOINT_PATH"] = ""
    from fastapi.testclient import TestClient

    from backend.app.main import app

    args.out.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as client:
        for name, path in ENDPOINTS.items():
            res = client.get(path)
            res.raise_for_status()
            body = res.json()
            if name == "health":  # describe the deploy truthfully: no server, no GPU
                body.update(model_state="not_configured", device="none (static demo)", cuda_available=False)
            (args.out / f"{name}.json").write_text(json.dumps(body, indent=2), encoding="utf8")
            print(f"{path} -> {(args.out / f'{name}.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

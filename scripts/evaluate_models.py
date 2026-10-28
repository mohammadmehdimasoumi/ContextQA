"""Compare BERT and DistilBERT on the same SQuAD 2.0 validation subset.

Usage (from the repo root):
    python scripts/evaluate_models.py --num-examples 300 --seed 42
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from contextqa.config import MODELS, PROJECT_ROOT  # noqa: E402
from contextqa.evaluation import (  # noqa: E402
    ModelReport, evaluate_engine, format_summary_table, load_squad_v2_examples, model_size_mb,
)
from contextqa.inference import QAEngine  # noqa: E402
from contextqa.utils import setup_logging  # noqa: E402

logger = logging.getLogger("evaluate_models")


def _hardware_info(device: str) -> dict:
    import torch
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-examples", type=int, default=200, help="subset size (default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    args = p.parse_args()
    setup_logging()

    examples = load_squad_v2_examples(args.num_examples, args.seed)
    n_unans = sum(not e["answers"] for e in examples)
    logger.info("Subset: %d examples (%d unanswerable), seed=%d", len(examples), n_unans, args.seed)

    reports: list[ModelReport] = []
    device = args.device
    for key in args.models:
        engine = QAEngine(key, device=args.device)
        device = engine.device
        size, params = model_size_mb(engine.model)
        scores, latency, skipped = evaluate_engine(engine, examples)
        reports.append(ModelReport(key, engine.spec.hf_id, len(examples), scores, latency,
                                   size, params, skipped))
        logger.info("%s -> EM %.1f | F1 %.1f", key, scores["EM"], scores["F1"])
        del engine  # free VRAM before loading the next model

    table = format_summary_table(reports)
    note = (f"Subset of {len(examples)} SQuAD 2.0 validation examples ({n_unans} unanswerable), "
            f"seed {args.seed}. Small subsets are NOT statistically representative of the full "
            "validation set. DistilBERT was trained on SQuAD 1.1 and cannot predict 'no answer', "
            "so its NoAns scores are ~0 by design. Size = fp32 weight bytes; latency is "
            "end-to-end per example (tokenize + forward + decode) on the hardware below.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps({
        "note": note, "hardware": _hardware_info(device),
        "reports": [dataclasses.asdict(r) for r in reports]}, indent=2))
    (args.output_dir / "results.md").write_text(table + "\n\n" + note + "\n")
    print("\n" + table + "\n\n" + note)


if __name__ == "__main__":
    main()

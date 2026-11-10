"""Evaluate CoReFusion outputs with the five metrics used in the study."""

import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir", type=Path, required=True, help="Infrared source image folder.")
    parser.add_argument("--vis", type=Path, required=True, help="Visible source image folder.")
    parser.add_argument("--fused", type=Path, required=True, help="Saved fused image folder.")
    parser.add_argument("--output", type=Path, default=Path("metrics.json"), help="JSON report path.")
    return parser.parse_args()


def main():
    args = parse_args()
    # Keep --help available before the optional metric dependencies are installed.
    try:
        import numpy as np
        import torch

        from corefusion.data import pair_paths
        from corefusion.metrics import METRIC_NAMES, evaluate_one
    except ImportError as error:
        raise SystemExit(
            "Evaluation requires the base environment and requirements-eval.txt. "
            "Use a torchvision build compatible with your installed PyTorch. "
            f"Import error: {error}"
        ) from error

    torch.set_num_threads(4)
    sources = pair_paths(args.ir, args.vis)
    fused = {name: fused_path for name, _, fused_path in pair_paths(args.ir, args.fused)}
    rows = []
    for index, (name, ir_path, vis_path) in enumerate(sources, start=1):
        values = evaluate_one(ir_path, vis_path, fused[name])
        rows.append({"name": name, **values})
        scores = " | ".join(f"{metric}={values[metric]:.6f}" for metric in METRIC_NAMES)
        print(f"[{index}/{len(sources)}] {name} | {scores}", flush=True)

    means = {
        metric: float(np.mean([row[metric] for row in rows]))
        for metric in METRIC_NAMES
    }
    undefined = [
        {"name": row["name"], "metrics": [key for key in METRIC_NAMES if not math.isfinite(row[key])]}
        for row in rows
        if any(not math.isfinite(row[key]) for key in METRIC_NAMES)
    ]

    def json_value(value):
        return value if math.isfinite(value) else None

    report = {
        "implementation": "CoReFusion study metrics (original numerical definitions)",
        "preprocessing": "PIL convert('L'); MI int32; SCD/VIF/Qabf/SSIM float32; no resizing",
        "count": len(rows),
        "higher_is_better": list(METRIC_NAMES),
        "per_image": [
            {"name": row["name"], **{key: json_value(row[key]) for key in METRIC_NAMES}}
            for row in rows
        ],
        "mean": {key: json_value(means[key]) for key in METRIC_NAMES},
        "undefined": undefined,
        "undefined_policy": "Nonfinite values are stored as null; affected metrics are not omitted from the mean.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Mean | " + " | ".join(f"{key}={means[key]:.6f}" for key in METRIC_NAMES))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()

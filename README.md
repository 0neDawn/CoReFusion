# CoReFusion

One-step infrared and visible image fusion.

[简体中文](README_zh.md)

## Release status

- [x] D1 model and inference code
- [x] Training modules and MSRS configuration
- [x] Evaluation: MI, SCD, VIF, Qabf, and SSIM
- [x] Paired examples from MSRS, M3FD, and FMB
- [ ] Standalone training entry point (`train.py`) — not released
- [ ] Pretrained checkpoints — coming soon

## Installation

Use Python 3.9 and run commands from the repository root. The dependencies pin PyTorch 2.6.0 and torchvision 0.21.0.

```bash
python -m pip install --index-url https://pypi.org/simple -r requirements.txt -r requirements-eval.txt
```

## Inference and evaluation

Checkpoints are not included. Place your checkpoint at `checkpoints/latest.pth` before running inference.

```bash
python test.py --device cpu
python evaluate.py \
  --ir data/MSRS/test/ir \
  --vis data/MSRS/test/vis \
  --fused results/MSRS/fused \
  --output results/MSRS/metrics.json
```

For a CUDA GPU, replace `--device cpu` with `--device cuda:0`. Inference saves color images to `fused/` and luminance images to `fused_y/`; evaluation above uses the color images. Inputs must be registered, have matching filename stems, and share the same dimensions.

If fused images already exist, run `evaluate.py` directly; no checkpoint is needed. Replace the input/output paths to evaluate M3FD, FMB, or your own data.

Inference and all five metrics were tested on the 15 bundled test pairs using a local checkpoint on CPU. This verifies the sample workflow, not full-dataset paper results.

See [sample data](data/README.md), [metric conventions](corefusion/metrics/README.md), and [attribution](NOTICE.md).

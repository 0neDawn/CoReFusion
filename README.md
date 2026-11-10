# CoReFusion

**CoReFusion**

This release contains the selected D1 architecture, its inference checkpoint, a training recipe for MSRS, and paired example images. CoReFusion predicts a latent state with one update, decodes coarse content, and reconstructs the output using selected source details at two scales. The detail heads combine local features, the predicted latent state, and spatial context; bounded residual compensation refines their weighted details.

## Installation

The source environment used Python 3.9.13, PyTorch 2.6.0 with CUDA 12.4, NumPy 1.26.4, and OpenCV 4.9.0. The training objective requires `torch.func.jvp`. A CUDA GPU is recommended for training and full-resolution inference.

Create an isolated Python environment and install the dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Choose the appropriate CUDA build of PyTorch for your system using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). The version in `requirements.txt` is the version used to prepare this release. Commands below assume the working directory is this repository.

## Fuse example images

The selected D1 weights are provided as `checkpoints/latest.pth`. See [checkpoint notes](checkpoints/README.md) for their provenance and supported use.

```bash
python test.py
python test.py --ir data/M3FD/test/ir --vis data/M3FD/test/vis --output results/M3FD
python test.py --ir data/FMB/test/ir --vis data/FMB/test/vis --output results/FMB
```

The default command processes `data/MSRS/test`. Each run writes color fusion images to `fused/`, fused luminance images to `fused_y/`, and per-image settings to `manifest.json` inside its output directory. Color outputs combine the predicted luminance with visible-image chrominance. Output filenames retain the source stems and use `.png`.

Inputs must be registered, have identical spatial dimensions, and have matching filename stems. Inference retains the input resolution; only the context thumbnail is resized, and temporary padding to a multiple of four is cropped from the output. Latent noise is seeded from `--seed` and the image stem, so changing the filename changes that noise realization. Existing results require a new output directory or `--overwrite`.

Use `--device cpu` for CPU execution. `--patch-batch` controls how many overlapping latent patches are processed together and can be reduced to lower memory use. One-step inference means one latent-state update; it still involves encoding, patch processing, and reconstruction. The elapsed times in the output manifest include image I/O and are not inference-latency benchmark measurements.

## Train on MSRS

The five bundled training pairs are sufficient for a pipeline demonstration. The deterministic split reserves one pair for validation and uses four for training. These samples are not sufficient to reproduce full-dataset results.

Run two optimization steps without changing the configured 50,000-step learning-rate schedule:

```bash
python train.py --stop-after 2 --batch-size 1 --output runs/smoke
```

Continue the same demonstration for two more steps:

```bash
python train.py --resume --stop-after 2 --batch-size 1 --output runs/smoke
```

For a full training run, obtain the complete MSRS training set and supply its directories:

```bash
python train.py --train-ir /path/to/MSRS/train/ir --train-vis /path/to/MSRS/train/vis --output runs/CoReFusion_MSRS --save-every 10000
```

The default recipe in `configs/corefusion_msrs.json` uses 50,000 steps, batch size 4, aligned 256 × 256 crops, synchronized horizontal flipping, and 128 × 128 full-image context thumbnails. AdamW uses weight decay 0.0001 and a base learning rate of 0.0001, with 500 warmup steps and cosine decay to 0.00001. The source-only fusion loss weights are 2, 4, and 1 for intensity, gradient, and structure, respectively. Additional reconstruction and shortcut terms are configured in the same file. No ground-truth fused image is required.

Training writes `latest.pth`, `best.pth`, periodic `step_XXXXXX.pth`, the run configuration, and JSON-lines logs. `best.pth` is selected by the held-out source-based fusion objective, not by test-set fusion metrics. Resume with the same configuration, data, code, and output directory, adding `--resume` to the original command. Full training checkpoints include optimizer and random-number-generator states. The bundled inference checkpoint does not include these states and cannot resume training. Use `--steps` only when intentionally changing the total optimization schedule; use `--stop-after` to pause a run while preserving that schedule.

## Evaluate fusion metrics

The optional evaluator reports MI, SCD, VIF, Qabf, and SSIM using the retained metric implementations:

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --ir data/MSRS/test/ir --vis data/MSRS/test/vis --fused results/MSRS/fused --output results/MSRS/metrics.json
```

Use the corresponding input and output directories for M3FD and FMB. Sample-set averages are demonstrations and should not be compared with paper results computed on the full evaluation sets. Keep the same fused-image representation and evaluator across compared methods.

## Repository contents

| Path | Purpose |
| --- | --- |
| `train.py`, `test.py`, `evaluate.py` | Training, fusion, and optional metric entry points |
| `configs/corefusion_msrs.json` | MSRS training recipe |
| `corefusion/model.py` | Source/context encoders, latent predictor, and reconstruction |
| `corefusion/objective.py`, `corefusion/losses.py` | Joint training objective and source-based losses |
| `corefusion/data.py`, `corefusion/color.py` | Pair loading, sampling, and color handling |
| `corefusion/training.py`, `corefusion/inference.py` | Training and full-resolution fusion routines |
| `corefusion/checkpoint.py` | State-dictionary loading and checkpoint saving |
| `checkpoints/latest.pth` | Selected D1 inference weights |
| `data/` | Five pairs per included split and sample hashes |

Only the D1 architecture is included. Experimental variants, dataset-specific server scripts, and earlier training outputs are excluded. See [NOTICE.md](NOTICE.md) for attribution and licensing status and [data/README.md](data/README.md) for sample selection.

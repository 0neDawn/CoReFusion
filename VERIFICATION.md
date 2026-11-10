# Release verification

The exported implementation was checked against the original D1 source and selected checkpoint on 2026-11-06, using Python 3.9.13, PyTorch 2.6.0+cu124, and an NVIDIA GeForce RTX 3090. TF32 was disabled, matching the original runner.

| Check | Result |
| --- | --- |
| Exported inference checkpoint | All 5,036,568 parameter values exactly match the selected original checkpoint; optimizer state removed |
| From-scratch initialization | Identical state tensors, parameter ordering, and RNG state at seed 2026 |
| Original versus released inference | Passed on a native-resolution pair from each of MSRS, M3FD, and FMB, plus an odd-sized MSRS crop |
| Saved images in the parity cases | RGB and luminance PNG files are byte-identical |
| Training preprocessing and sampling | Exact matches for deterministic crops and three seeded random-sampling cases |
| Joint objective and gradients | Match within FP32 verification tolerances; gradient tolerance was atol 2e-6 / rtol 1e-4 |
| Standalone GPU test entry point | All 15 bundled test pairs run successfully and retain their input dimensions |
| Training and checkpoint resume | Two steps from scratch and one step plus resume both pass; maximum parameter difference 1.87e-9, final CPU/CUDA RNG states equal |
| CPU inference | One native-resolution MSRS sample passes |
| Five evaluation metrics | All 75 scores on the 15 test pairs exactly match the original metric routines in the same environment |
| Sample file integrity | All 40 images match the SHA-256 hashes in data/manifest.json |

Metric verification used SciPy 1.13.1, Pillow 11.2.1, and torchvision 0.21.0+cu124. Metric function definitions were also checked against the supplied originals by comparing their Python syntax trees.

These checks validate the exported code and short training/resume workflow. A new 50,000-step training run was not performed when building this release, and the bundled sample-set results are not full-benchmark results.

The archive includes `release_manifest.json` with file sizes and SHA-256 hashes. The separate `CoReFusion.zip.sha256` file identifies the distributed ZIP archive.

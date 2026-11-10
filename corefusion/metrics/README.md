# Evaluation implementation

This optional evaluator retains the five metrics used by the CoReFusion
experiments: MI, SCD, VIF, Qabf, and SSIM. The function bodies were extracted
without numerical changes from the authors' supplied `Metric.py`, `Qabf.py`,
and `ssim.py`. Unused metric functions and spreadsheet output code were removed.
These are existing evaluation routines, not new metrics introduced by CoReFusion.

The retained dependencies are:

- `Metric.py`: `Hab`, `MI_function`, `corr2`, `SCD_function`,
  `fspecial_gaussian`, `vifp_mscale`, `VIF_function`, `Qabf_function`,
  and `SSIM_function`.
- `Qabf.py`: `get_Qabf`, including its nested helpers.
- `ssim.py`: `_fspecial_gauss_1d`, `gaussian_filter`, `_ssim`, and `ssim`.

The source-file SHA-256 hashes used for extraction are:

| Source | SHA-256 |
| --- | --- |
| `Metric.py` | `8fb4f82ecd7f284e02e786bb659cba0511956020795d1f6d637210ae27402be0` |
| `Qabf.py` | `d004e381dd3af42505881fc48a729b7ca0a30157caeffebf524c7ad422d6f329` |
| `ssim.py` | `d1ab667546a328537e71a4025815c1babb62bed8f92e5bb1b2181e56b23a22dc` |

## Input and numerical conventions

Each source and fused image is opened with Pillow and converted with
`convert('L')`, exactly as in the original evaluation entrypoint. MI receives
`int32` arrays. The other four metrics receive `float32` arrays, with values
in `[0, 255]`. No resize, alignment, per-image normalization, or histogram
adjustment is applied. Evaluation uses the saved color fused output when
reproducing the study, not the separately saved fused luminance output.

The original SSIM routine uses `torchvision.transforms.functional.to_tensor`
followed by multiplication by `255.0`, including when the input is a float32
array. This convention is retained to reproduce the reported experiment.
SSIM and VIF add the two source-to-fused scores. Consequently these fusion
metrics need not be bounded by 1. Their values should not be compared directly
with differently implemented or averaged metrics.

Strict stem matching is used across infrared, visible, and fused directories.
Duplicate or unmatched names and inconsistent dimensions are errors. Images
are not silently discarded. All per-image values and arithmetic dataset means
are stored at full precision. For degenerate inputs, nonfinite results are
represented as JSON `null` and explicitly listed; such samples are not silently
excluded from means.

Install the optional packages with `requirements-eval.txt` after selecting a
torchvision build compatible with the already installed PyTorch environment.
Run `python evaluate.py --help` for the required source/output directory flags.

## Verification

The retained function definitions were checked for AST equality with the source
definitions. Runtime validation on all 15 bundled test pairs produced exactly
the same 75 scores as the original routines in the same environment, using the
Pillow conversion and dtypes above. The verified optional dependency versions
are pinned in `requirements-eval.txt`. See the root `VERIFICATION.md` for the
release checks. No replacement SSIM library or metric correction was introduced.

The supplied source files did not contain a license or upstream attribution
header. Their upstream redistribution terms should be verified by the release
authors before a public repository license is applied to these files.

"""Five fusion metrics using the original CoReFusion study implementation."""

import numpy as np
from PIL import Image

from ._algorithms import (
    MI_function,
    Qabf_function,
    SCD_function,
    SSIM_function,
    VIF_function,
)

METRIC_NAMES = ("MI", "SCD", "VIF", "Qabf", "SSIM")


def evaluate_one(ir_path, vis_path, fused_path):
    """Evaluate one pair using the study's PIL grayscale conversion and dtypes.

    MI consumes int32 arrays; the other four metrics consume float32 arrays
    with grayscale values in [0, 255]. Arrays are never resized or normalized.
    """
    with Image.open(fused_path) as image:
        f_img = image.convert("L")
    with Image.open(ir_path) as image:
        ir_img = image.convert("L")
    with Image.open(vis_path) as image:
        vi_img = image.convert("L")

    f_img_int = np.array(f_img).astype(np.int32)
    f_img_double = np.array(f_img).astype(np.float32)
    ir_img_int = np.array(ir_img).astype(np.int32)
    ir_img_double = np.array(ir_img).astype(np.float32)
    vi_img_int = np.array(vi_img).astype(np.int32)
    vi_img_double = np.array(vi_img).astype(np.float32)

    if not (ir_img_int.shape == vi_img_int.shape == f_img_int.shape):
        raise ValueError(
            f"Metric inputs must have matching dimensions: IR {ir_img_int.shape}, "
            f"VIS {vi_img_int.shape}, fused {f_img_int.shape}; fused file {fused_path}"
        )

    # The original script suppresses warnings, including division warnings in
    # flat regions. Keep those numerical results; do not add stabilizing terms.
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {
            "MI": MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256),
            "SCD": SCD_function(ir_img_double, vi_img_double, f_img_double),
            "VIF": VIF_function(ir_img_double, vi_img_double, f_img_double),
            "Qabf": Qabf_function(ir_img_double, vi_img_double, f_img_double),
            "SSIM": SSIM_function(ir_img_double, vi_img_double, f_img_double),
        }
    return {name: float(values[name]) for name in METRIC_NAMES}


__all__ = ["METRIC_NAMES", "evaluate_one"]

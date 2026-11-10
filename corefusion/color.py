"""Visible chrominance reconstruction for CoReFusion outputs."""

import math

import torch


def luminance(rgb):
    """Return luminance for a floating-point RGB tensor of shape B x 3 x H x W."""
    return (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(
        1, keepdim=True
    )


def restore_visible_color(fused_y, visible_rgb, strength=1.0):
    """Combine fused luminance with visible chrominance within the RGB gamut.

    A shared chrominance scale is reduced only where necessary to keep RGB in
    [0, 1], preserving luminance and hue up to numerical precision. This does
    not infer the physical color of objects observable only in infrared.
    """
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color strength must be finite and in [0, 1].")
    if (
        fused_y.ndim != 4
        or visible_rgb.ndim != 4
        or fused_y.shape[1] != 1
        or visible_rgb.shape[1] != 3
    ):
        raise ValueError("Expected B x 1 x H x W luminance and B x 3 x H x W visible RGB.")
    if (
        fused_y.shape[0] != visible_rgb.shape[0]
        or fused_y.shape[-2:] != visible_rgb.shape[-2:]
    ):
        raise ValueError("Fused luminance and visible RGB must be spatially registered.")
    y = fused_y.clamp(0, 1)
    chroma = (visible_rgb.clamp(0, 1) - luminance(visible_rgb.clamp(0, 1))) * strength
    eps = torch.finfo(chroma.dtype).eps
    positive_bound = (1 - y) / chroma.clamp_min(eps)
    negative_bound = y / (-chroma).clamp_min(eps)
    bounds = torch.where(
        chroma > eps,
        positive_bound,
        torch.where(chroma < -eps, negative_bound, torch.ones_like(chroma)),
    )
    scale = bounds.amin(dim=1, keepdim=True).clamp(0, 1)
    return (y + scale * chroma).clamp(0, 1)

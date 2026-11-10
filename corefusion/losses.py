"""Source-only reconstruction and fusion losses used by CoReFusion."""

import math

import torch
from torch.nn import functional as F


def sobel(x):
    kernel = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 4
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return (
        F.conv2d(padded, kernel[None, None]),
        F.conv2d(padded, kernel.t()[None, None]),
    )


def laplacian(x):
    kernel = x.new_tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) / 4
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), kernel[None, None])


def ssim_map(x, y):
    """Return the SSIM loss map (1 - SSIM) over valid 7x7 uniform windows."""
    def pool(value):
        return F.avg_pool2d(value, 7, 1, 0)

    mean_x, mean_y = pool(x), pool(y)
    variance_x = (pool(x * x) - mean_x * mean_x).clamp_min(0)
    variance_y = (pool(y * y) - mean_y * mean_y).clamp_min(0)
    covariance = pool(x * y) - mean_x * mean_y
    covariance = torch.where(
        (variance_x > 0) & (variance_y > 0), covariance, torch.zeros_like(covariance)
    )
    similarity = (
        (2 * mean_x * mean_y + 0.01**2) * (2 * covariance + 0.03**2)
    ) / (
        (mean_x.square() + mean_y.square() + 0.01**2)
        * (variance_x + variance_y + 0.03**2)
    )
    return 1 - similarity


def ssim_per_image(x, y):
    return ssim_map(x, y).flatten(1).mean(1)


def reconstruction(x, y, weights):
    pixel = F.l1_loss(x, y)
    edge = sum(F.l1_loss(a, b) for a, b in zip(sobel(x), sobel(y)))
    high = F.l1_loss(laplacian(x), laplacian(y))
    structure = ssim_per_image(x, y).mean()
    total = (
        weights["pixel"] * pixel + weights["edge"] * edge
        + weights["high"] * high + weights["structure"] * structure
    )
    return total, dict(pixel=pixel, edge=edge, high=high, structure=structure)


def fusion(
    x, ir, vis, weights, per_image=False,
    edge_mode="legacy", structure_mode="local", local_gamma=1.5,
):
    """Compute D1 intensity, gradient-magnitude, and local structure losses.

    Mode arguments preserve the released configuration interface. Alternative
    objectives are intentionally excluded from this single-model release.
    """
    if edge_mode != "legacy" or structure_mode != "local":
        raise ValueError("CoReFusion uses legacy gradient and local structure loss modes")
    if not math.isfinite(local_gamma) or local_gamma <= 0:
        raise ValueError("local_gamma must be finite and positive")

    intensity = (x - torch.maximum(ir, vis)).abs().flatten(1).mean(1)
    gradient_ir, gradient_vis, gradient_fused = sobel(ir), sobel(vis), sobel(x)
    edge = sum(
        (a.abs() - torch.maximum(b.abs(), c.abs())).abs().flatten(1).mean(1)
        for a, b, c in zip(gradient_fused, gradient_ir, gradient_vis)
    )

    energy_ir = F.avg_pool2d(sum(g.abs() for g in gradient_ir), 7, 1, 0)
    energy_vis = F.avg_pool2d(sum(g.abs() for g in gradient_vis), 7, 1, 0)
    weight_ir = torch.where(
        energy_ir + energy_vis > 1e-8,
        energy_ir / (energy_ir + energy_vis).clamp_min(1e-8),
        torch.full_like(energy_ir, 0.5),
    ).detach()
    if local_gamma != 1.0:
        power_ir = (energy_ir + 1e-8).pow(local_gamma)
        power_vis = (energy_vis + 1e-8).pow(local_gamma)
        weight_ir = torch.where(
            energy_ir + energy_vis > 1e-8,
            power_ir / (power_ir + power_vis).clamp_min(1e-30),
            torch.full_like(energy_ir, 0.5),
        ).detach()
    structure = (
        weight_ir * ssim_map(x, ir) + (1 - weight_ir) * ssim_map(x, vis)
    ).flatten(1).mean(1)

    total = (
        weights["intensity"] * intensity + weights["edge"] * edge
        + weights["structure"] * structure
    )
    return (total if per_image else total.mean()), dict(
        intensity=intensity.mean(), edge=edge.mean(), structure=structure.mean()
    )

"""Joint source-only objective for the released CoReFusion model.

The continuous-time shortcut objective is adapted from ExplicitShortCut
(ICLR 2026; official loss/esc_loss.py). CoReFusion uses an empirical endpoint
distribution over the two sources of the same registered pair, instead of
unrelated batch images. No ground-truth fused image is assumed.
"""

import math

import torch
from torch.func import jvp
from torch.nn import functional as F

from .losses import fusion, reconstruction


def detach_cache(cache):
    """Detach only the cache entries needed by latent-velocity prediction."""
    return {
        key: tuple(x.detach() for x in value) if isinstance(value, tuple) else value.detach()
        for key, value in cache.items()
        if key in ("zi", "zv", "fi", "fv", "tokens", "whole", "xy")
    }


def times(cfg, step, batch_size, device):
    # One shared time per batch selects a single patch geometry for this loss.
    values = (
        torch.randn(2, device=device) * cfg.TIME_SIGMA + cfg.TIME_MU
    ).sigmoid().sort().values
    r, t = values[0], values[1]
    progress = min(step / cfg.TIME_SAMPLER_TRANSITION, 1.0)
    # Match the official shortcut implementation's arccos transition.
    if torch.rand((), device=device) < math.acos(progress) / (math.pi / 2):
        r = torch.zeros_like(r)
    if torch.rand((), device=device) < cfg.EQUAL_TIME_PROB:
        r = t
    return r.expand(batch_size).clone(), t.expand(batch_size).clone()


def plugin_velocity(xt, source_pair, t):
    """Empirical posterior velocity for IR/VIS endpoints of one registered pair."""
    time = t[:, None, None, None, None].clamp_min(1e-4)
    epsilon = (xt[:, None] - (1 - time) * source_pair) / time
    log_probability = -0.5 * epsilon.square().flatten(2).sum(-1)
    weights = log_probability.softmax(1)
    velocity = ((epsilon - source_pair) * weights[:, :, None, None, None]).sum(1)
    return velocity, weights


def esc_loss(model, cache, cfg, step):
    sources = torch.stack([cache["zi"].detach(), cache["zv"].detach()], 1)
    batch_size = len(sources)
    device = sources.device
    index = torch.randint(2, (batch_size,), device=device)
    target = sources[torch.arange(batch_size, device=device), index]
    r, t = times(cfg, step, batch_size, device)
    noise = torch.randn_like(target)
    time = t[:, None, None, None]
    xt = ((1 - time) * target + time * noise).detach()
    velocity = noise - target
    posterior, weights = plugin_velocity(xt, sources, t)
    choose = (torch.rand(batch_size, device=device) < cfg.PLUGIN_PROB)[:, None, None, None]
    velocity = torch.where(choose, posterior, velocity).detach()
    patch, stride = model.geometry(t, cfg.PATCH_SIZES, cfg.PATCH_STRIDES)
    frozen = detach_cache(cache)

    with torch.no_grad():
        def field(z, r_value, t_value):
            return model.velocity(z, r_value, t_value, frozen, patch, stride, cfg.PATCH_BATCH)

        _, derivative = jvp(
            field, (xt, r, t),
            (velocity, torch.zeros_like(r), torch.ones_like(t)),
        )
        warmup = min(step / cfg.TANGENT_WARMUP, 1.0)
        target_velocity = (velocity - (t - r)[:, None, None, None] * derivative * warmup).detach()

    prediction = model.velocity(xt, r, t, cache, patch, stride, cfg.PATCH_BATCH)
    error = (prediction - target_velocity).square().flatten(1).mean(1)
    logvar = model.field.logvar(torch.stack([r, t], 1)).flatten().clamp(-6, 6)
    weighted = (error * torch.exp(-logvar) + logvar).mean()
    return weighted, dict(
        esc_mse=error.mean(), esc_weighted=weighted, t=t.mean(), r=r.mean(),
        patch=target.new_tensor(patch), stride=target.new_tensor(stride),
        tangent_rms=derivative.square().mean().sqrt(), plugin_ir=weights[:, 0].mean(),
        logvar=logvar.mean(),
    )


def loss(model, batch, cfg, step):
    ir, vis, whole, xy = batch
    cache = model.cache(ir, vis, whole, xy)
    fused, _ = model.one_step(cache, cfg)
    energy, parts = fusion(fused, ir, vis, cfg.FUSION_WEIGHTS, **cfg.LOSS_MODES)

    source = torch.cat([ir, vis])
    latent = torch.cat([cache["zi"], cache["zv"]])
    reconstruction_loss, _ = reconstruction(model.codec.decode(latent), source, cfg.AE_WEIGHTS)
    source_power = F.pixel_unshuffle(source, 4).square().flatten(1).mean(1).detach()
    latent_power = latent.square().flatten(1).mean(1)
    scale = ((latent_power - source_power) / (source_power + 0.01)).square().mean()
    shortcut, details = esc_loss(model, cache, cfg, step)
    details["esc_enabled"] = fused.new_tensor(1.0)
    total = (
        energy + cfg.RECON_WEIGHT * reconstruction_loss
        + cfg.ESC_WEIGHT * shortcut + cfg.LATENT_SCALE_WEIGHT * scale
    )
    return total, dict(
        fusion=energy, reconstruction=reconstruction_loss, latent_scale=scale,
        **parts, **details,
    )

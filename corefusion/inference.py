"""Full-resolution fusion with deterministic, filename-derived latent noise."""

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .checkpoint import load_checkpoint
from .color import restore_visible_color
from .config import configure_runtime
from .data import context, coordinates, pair_paths, read, save_tensor


def fuse_pair(model, cfg, infrared_path, visible_path, name, device, seed=2026):
    infrared, visible, rgb = read(infrared_path, visible_path)
    height, width = infrared.shape[-2:]
    whole = context(infrared, visible, cfg.GLOBAL_SIZE)[None].to(device)
    padded_h, padded_w = (height + 3) // 4 * 4, (width + 3) // 4 * 4
    xy = coordinates(height, width, 0, 0, padded_h, padded_w)[None].to(device)
    padding = (0, padded_w - width, 0, padded_h - height)
    infrared = F.pad(infrared[None].to(device), padding, mode="replicate")
    visible = F.pad(visible[None].to(device), padding, mode="replicate")
    rgb = rgb[None].to(device)
    noise_seed = int(hashlib.sha256((str(seed) + name).encode()).hexdigest()[:8], 16)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    noise = torch.randn((1, 16, padded_h // 4, padded_w // 4), device=device, generator=generator)
    with torch.inference_mode():
        fused, _ = model.one_step(model.cache(infrared, visible, whole, xy), cfg, noise=noise)
        fused = fused[..., :height, :width].clamp(0, 1)
        return fused, restore_visible_color(fused, rgb), noise_seed


def run(checkpoint, infrared_dir, visible_dir, output, device="cuda:0", patch_batch=16,
        seed=2026, limit=0, overwrite=False):
    if patch_batch < 1 or limit < 0:
        raise ValueError("Patch batch must be positive and limit nonnegative.")
    device = configure_runtime(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval().requires_grad_(False)
    cfg = SimpleNamespace(**state["config"])
    cfg.PATCH_BATCH = patch_batch
    pairs = pair_paths(infrared_dir, visible_dir)
    if limit:
        pairs = pairs[:limit]
    output = Path(output)
    if not overwrite:
        conflicts = [output / folder / f"{name}.png" for name, _, _ in pairs
                     for folder in ("fused", "fused_y") if (output / folder / f"{name}.png").exists()]
        if conflicts or (output / "manifest.json").exists():
            raise FileExistsError(f"Existing results in {output}; choose another --output or use --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, infrared_path, visible_path in pairs:
        tick = time.perf_counter()
        y, color, noise_seed = fuse_pair(model, cfg, infrared_path, visible_path, name, device, seed)
        save_tensor(output / "fused" / f"{name}.png", color)
        save_tensor(output / "fused_y" / f"{name}.png", y)
        row = dict(name=name, height=y.shape[-2], width=y.shape[-1], noise_seed=noise_seed,
                   elapsed_seconds=time.perf_counter() - tick)
        rows.append(row)
        print(json.dumps(row), flush=True)
    manifest = dict(method="CoReFusion", training_step=state["step"],
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    seed=seed, patch_batch=patch_batch, state_updates=1, rows=rows,
                    timing_note="Per-pair elapsed time includes preprocessing and file I/O; not a latency benchmark.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output

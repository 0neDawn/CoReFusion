"""From-scratch training and exact continuation of CoReFusion training runs."""

import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .checkpoint import atomic_save, load_checkpoint
from .config import config_dict, configure_runtime
from .data import Sources
from .losses import fusion
from .model import CoReFusion
from .objective import loss


def source_hashes():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}


def validate(model, source, cfg):
    model.eval()
    device = torch.device(cfg.DEVICE)
    devices = [device.index or 0] if device.type == "cuda" else []
    scores = []
    with torch.random.fork_rng(devices=devices), torch.inference_mode():
        torch.manual_seed(cfg.SEED + 17)
        for path in source.val[:cfg.VALIDATION_IMAGES]:
            batch = [x[None].to(device) for x in source.one(path, random=False)]
            infrared, visible, _, _ = batch
            image, _ = model.one_step(model.cache(*batch), cfg)
            score, _ = fusion(image, infrared, visible, cfg.FUSION_WEIGHTS, **cfg.LOSS_MODES)
            scores.append(float(score))
    model.train()
    return sum(scores) / len(scores)


def train(cfg, resume=False, stop_after=None):
    """``stop_after`` pauses after this many new steps without changing the schedule."""
    if cfg.STEPS < 1 or cfg.BATCH_SIZE < 1 or cfg.SAVE_EVERY < 1:
        raise ValueError("Steps, batch size and checkpoint interval must be positive.")
    if cfg.CROP_SIZE < 4 or cfg.CROP_SIZE % 4 or cfg.GLOBAL_SIZE % 8:
        raise ValueError("Crop size must be divisible by 4 and context size by 8.")
    if len(cfg.PATCH_SIZES) != 3 or len(cfg.PATCH_STRIDES) != 3:
        raise ValueError("Exactly three patch sizes and strides are required.")
    if any(p < 2 or p % 2 or s < 1 or s > p for p, s in zip(cfg.PATCH_SIZES, cfg.PATCH_STRIDES)):
        raise ValueError("Patch sizes must be positive even integers; strides must cover every pixel.")
    if stop_after is not None and stop_after < 1:
        raise ValueError("--stop-after must be positive.")
    device = configure_runtime(cfg.DEVICE)
    torch.manual_seed(cfg.SEED)
    root = Path(cfg.RUN_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    latest = root / "latest.pth"
    if latest.exists() and not resume:
        raise FileExistsError(f"Existing run: {root}. Use --resume or another --output.")
    if resume and not latest.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {latest}")
    source = Sources(cfg)
    model = CoReFusion(cfg.WIDTH).to(device)
    torch.manual_seed(cfg.SEED + 101)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    start, best = 0, float("inf")
    current = config_dict(cfg)
    hashes = source_hashes()
    if resume:
        model, state = load_checkpoint(latest, device)
        if state["format"] != "CoReFusion-1" or "optimizer" not in state:
            raise ValueError("Resume requires a full checkpoint saved by train.py, not the inference weights.")
        allowed = {"DEVICE", "RUN_ROOT", "TRAIN_IR", "TRAIN_VIS"}
        for key, value in state["config"].items():
            if key not in allowed and current.get(key) != value:
                raise ValueError(f"Resume configuration differs: {key}")
        if state["source_hashes"] != hashes:
            raise ValueError("Training source files changed since this checkpoint.")
        if state["train_names"] != [p[0] for p in source.paths] or state["val_names"] != [p[0] for p in source.val]:
            raise ValueError("Training/validation filenames changed since this checkpoint.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        optimizer.load_state_dict(state["optimizer"])
        start, best = state["step"], state["best"]
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda"):
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    if start >= cfg.STEPS:
        print(f"Training already complete at step {start}.", flush=True)
        return latest
    metadata = dict(format="CoReFusion-1", spec=dict(width=cfg.WIDTH), config=current,
                    train_names=[p[0] for p in source.paths], val_names=[p[0] for p in source.val],
                    source_hashes=hashes)
    (root / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    last_step = min(cfg.STEPS, start + stop_after) if stop_after else cfg.STEPS
    print(json.dumps(dict(start=start, steps=cfg.STEPS, stop_at=last_step,
                          parameters=sum(p.numel() for p in model.parameters()),
                          training_pairs=len(source.paths), validation_pairs=len(source.val))), flush=True)
    tick = time.perf_counter()
    model.train()
    for step in range(start + 1, last_step + 1):
        factor = cfg.MIN_LR_RATIO + (1 - cfg.MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * step / cfg.STEPS))
        factor *= min(step / cfg.LR_WARMUP, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = cfg.LEARNING_RATE * factor
        optimizer.zero_grad(set_to_none=True)
        value, terms = loss(model, source.batch(cfg.BATCH_SIZE, device), cfg, step)
        if not torch.isfinite(value):
            raise FloatingPointError(f"Nonfinite training loss at step {step}.")
        value.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == start + 1 or step % 25 == 0 or step == last_step:
            row = dict(step=step, total=float(value.detach()), grad_norm=float(norm),
                       lr=optimizer.param_groups[0]["lr"], elapsed_seconds=time.perf_counter() - tick,
                       **{key: float(val.detach()) for key, val in terms.items()})
            print(json.dumps(row), flush=True)
            with (root / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
        if step % cfg.SAVE_EVERY == 0 or step == last_step:
            score = validate(model, source, cfg)
            improved = score < best
            best = min(score, best)
            record = dict(metadata, step=step, best=best, model=model.state_dict(),
                          optimizer=optimizer.state_dict(), rng_cpu=torch.get_rng_state(),
                          rng_cuda=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
            atomic_save(record, latest)
            if step % cfg.SAVE_EVERY == 0 or step == cfg.STEPS:
                atomic_save(record, root / f"step_{step:06d}.pth")
            if improved:
                atomic_save(record, root / "best.pth")
            with (root / "validation.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(step=step, source_objective=score, best=best)) + "\n")
            print(json.dumps(dict(step=step, validation=score, checkpoint=str(latest))), flush=True)
    return latest

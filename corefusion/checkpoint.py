"""State-dict checkpoints without serialized model objects."""

from pathlib import Path

import torch

from .model import CoReFusion


def load_checkpoint(path, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    kind = state.get("format")
    if kind not in {"CoReFusion-1", "PLPFusion-D012-1"}:
        raise ValueError(f"Unsupported CoReFusion checkpoint format: {kind}")
    if kind == "PLPFusion-D012-1" and state["spec"].get("variant") != "D1":
        raise ValueError("Only the original D1 architecture is supported.")
    model = CoReFusion(width=state["spec"]["width"])
    model.load_state_dict(state["model"], strict=True)
    return model.to(device), state


def atomic_save(record, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(record, temporary)
    temporary.replace(path)

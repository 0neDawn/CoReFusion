"""Portable configuration for the released CoReFusion architecture."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch


def load_config(path):
    return SimpleNamespace(**json.loads(Path(path).read_text(encoding="utf-8")))


def config_dict(config):
    return json.loads(json.dumps(vars(config), default=str))


def configure_runtime(device):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch or use --device cpu.")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device

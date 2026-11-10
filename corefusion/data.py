"""Registered image pairs, crop sampling, and image I/O for CoReFusion."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .color import luminance

EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_index(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    items = {}
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            if path.stem in items:
                raise ValueError(
                    f"Duplicate image stem {path.stem!r}: {items[path.stem]} and {path}. "
                    "Each source folder must contain one image per stem."
                )
            items[path.stem] = path
    if not items:
        raise ValueError(f"No supported images found in {folder}")
    return items


def pair_paths(ir_dir, vis_dir):
    """Return (stem, infrared path, visible path) tuples sorted by exact stem.

    Pairing is strict: both folders must have the same set of stems. Extensions
    can differ, but unmatched images and duplicate stems are rejected.
    """
    ir, vis = _image_index(ir_dir), _image_index(vis_dir)
    if ir.keys() != vis.keys():
        ir_only = sorted(ir.keys() - vis.keys())
        vis_only = sorted(vis.keys() - ir.keys())
        raise ValueError(
            f"Unmatched image stems between {ir_dir} and {vis_dir}: "
            f"IR-only ({len(ir_only)})={ir_only[:10]}, "
            f"VIS-only ({len(vis_only)})={vis_only[:10]}"
        )
    return [(name, ir[name], vis[name]) for name in sorted(ir)]


def _read_image(path):
    # imdecode supports Unicode paths on Windows.
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Image does not exist: {path}")
    encoded = np.fromfile(path, dtype=np.uint8)
    if not encoded.size:
        raise ValueError(f"Image file is empty: {path}")
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    if image.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"Only unsigned 8/16-bit images are supported: {path} ({image.dtype})")
    return image


def read(ir_path, vis_path):
    """Return IR, visible luminance, and visible RGB as CPU tensors in [0, 1].

    Sources retain their original spatial dimensions; no alignment or resizing
    is performed. Shapes are respectively 1 x H x W, 1 x H x W, and 3 x H x W.
    """
    ir, vis = _read_image(ir_path), _read_image(vis_path)
    if ir.ndim == 3:
        if ir.shape[2] != 3 or not (
            np.array_equal(ir[..., 0], ir[..., 1])
            and np.array_equal(ir[..., 1], ir[..., 2])
        ):
            raise ValueError(f"IR must be grayscale, not pseudocolor or RGBA: {ir_path}")
        ir = ir[..., 0]
    if ir.ndim != 2:
        raise ValueError(f"IR must have two spatial dimensions: {ir_path} ({ir.shape})")
    ir = ir.astype(np.float32) / (65535 if ir.dtype == np.uint16 else 255)
    if vis.dtype != np.uint8:
        raise ValueError(f"Visible images must be unsigned 8-bit: {vis_path} ({vis.dtype})")
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    if vis.ndim != 3 or vis.shape[2] != 3:
        raise ValueError(f"Visible images must be grayscale or three-channel, without alpha: {vis_path}")
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if ir.shape != vis.shape[:2] or min(ir.shape) < 2:
        raise ValueError(
            "Require registered, same-size images with both dimensions at least 2: "
            f"{ir_path} {ir.shape}; {vis_path} {vis.shape}"
        )
    ir = torch.from_numpy(ir.copy())[None]
    rgb = torch.from_numpy(vis.copy()).permute(2, 0, 1).float() / 255
    return ir, luminance(rgb[None])[0], rgb


def save_tensor(path, tensor):
    """Write one 1 x C x H x W luminance or RGB tensor to an 8-bit PNG file."""
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] not in (1, 3):
        raise ValueError(f"Expected 1 x 1/3 x H x W image tensor, got {tuple(tensor.shape)}")
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError(f"Output images use PNG format; choose a .png path: {path}")
    array = tensor.detach().clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    array = (array * 255).round().astype(np.uint8)
    if array.shape[2] == 1:
        array = array[..., 0]
    else:
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise RuntimeError(f"Failed encoding {path}")
    buffer.tofile(path)


def context(ir, vi, size):
    """Resize the concatenated source intensities to the global thumbnail."""
    return F.interpolate(
        torch.cat([ir, vi])[None],
        size=(size, size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )[0]


def coordinates(h, w, top, left, hh, ww):
    """Create the original normalized coordinate grid at latent resolution."""
    y, x = torch.meshgrid(
        torch.linspace(2 * top / h - 1, 2 * (top + hh) / h - 1, hh // 4),
        torch.linspace(2 * left / w - 1, 2 * (left + ww) / w - 1, ww // 4),
        indexing="ij",
    )
    return torch.stack([x, y])


class Sources:
    """Deterministic source split and synchronized random crop/flip sampling."""

    def __init__(self, cfg):
        paths = pair_paths(cfg.TRAIN_IR, cfg.TRAIN_VIS)
        paths = sorted(
            paths,
            key=lambda p: hashlib.sha256((str(cfg.SEED) + p[0]).encode()).hexdigest(),
        )
        if len(paths) < 3:
            raise ValueError("At least three registered training image pairs are required.")
        if not 0 < cfg.VALIDATION_FRACTION < 1:
            raise ValueError("VALIDATION_FRACTION must lie strictly between 0 and 1.")
        n = max(1, int(len(paths) * cfg.VALIDATION_FRACTION))
        self.val, self.paths = paths[:n], paths[n:]
        self.crop = cfg.CROP_SIZE
        self.size = cfg.GLOBAL_SIZE
        self.cache = {}

    def one(self, path, random=True):
        name, a, b = path
        if name not in self.cache:
            ir, vi, _ = read(a, b)
            if len(self.cache) >= 64:
                self.cache.pop(next(iter(self.cache)))
            self.cache[name] = (ir, vi, context(ir, vi, self.size))
        ir, vi, whole = self.cache[name]
        h, w = ir.shape[-2:]
        s = self.crop
        if min(h, w) < s:
            raise ValueError(f"Crop size {s} exceeds source size {h} x {w} for {name}.")
        top = int(torch.randint(h - s + 1, ())) if random else (h - s) // 2
        left = int(torch.randint(w - s + 1, ())) if random else (w - s) // 2
        xy = coordinates(h, w, top, left, s, s)
        a = ir[:, top : top + s, left : left + s]
        b = vi[:, top : top + s, left : left + s]
        if random and torch.rand(()) < 0.5:
            a, b, whole, xy = a.flip(-1), b.flip(-1), whole.flip(-1), xy.flip(-1).clone()
            xy[0] = -xy[0]
        return a, b, whole, xy

    def batch(self, n, device):
        rows = [self.one(self.paths[int(i)]) for i in torch.randint(len(self.paths), (n,))]
        return [torch.stack([row[j] for row in rows]).to(device) for j in range(4)]

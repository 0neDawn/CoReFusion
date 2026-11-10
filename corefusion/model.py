"""CoReFusion: one-step fusion with decoupled source-detail reconstruction.

This module contains the released D1 architecture only. Parameter names and
initialization order are retained for compatibility with the trained weights.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def up(x, target):
    """Bilinearly resize ``x`` to the spatial dimensions of ``target``."""
    return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)


def pyramid(x):
    """Return the quarter-resolution base and two source-detail bands."""
    half = F.avg_pool2d(x, 2)
    base = F.avg_pool2d(half, 2)
    return base, (x - up(half, x), half - up(base, half))


def reconstruct(base, details):
    """Reconstruct an image from its base and full/half-resolution details."""
    half = up(base, details[1]) + details[1]
    return up(half, details[0]) + details[0]


class Block(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + 0.2 * self.net(x)


class Codec(nn.Module):
    """Learnable 4x spatial transform with a full 16-channel DCT initialization."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(16, 16, 1, bias=False)
        # Reserve the original parameter traversal order for optimizer resumes.
        self.er = nn.Identity()
        self.dr = nn.Identity()

        # The training initializer consumed these draws before constructing dec.
        # Preserve that RNG sequence without retaining obsolete parameters.
        for in_channels, out_channels in ((16, 32), (32, 16), (16, 32), (32, 16)):
            nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.dec = nn.Conv2d(16, 16, 1, bias=False)
        frequencies = sorted(
            [(u, v) for u in range(4) for v in range(4)],
            key=lambda uv: (sum(uv), uv[0]),
        )
        basis = []
        for u, v in frequencies:
            a = 0.5 if u == 0 else math.sqrt(0.5)
            b = 0.5 if v == 0 else math.sqrt(0.5)
            basis.append(
                [
                    a * b * math.cos(math.pi * (2 * x + 1) * u / 8)
                    * math.cos(math.pi * (2 * y + 1) * v / 8)
                    for x in range(4) for y in range(4)
                ]
            )
        with torch.no_grad():
            weights = torch.tensor(basis)
            self.enc.weight.copy_(weights[:, :, None, None])
            self.dec.weight.copy_(weights.t()[:, :, None, None])

        self.er = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1), nn.SiLU(), Block(64),
            nn.Conv2d(64, 16, 3, padding=1),
        )
        self.dr = nn.Sequential(
            nn.Conv2d(16, 64, 3, padding=1), nn.SiLU(), Block(64),
            nn.Conv2d(64, 16, 3, padding=1),
        )
        for part in (self.er, self.dr):
            nn.init.zeros_(part[-1].weight)
            nn.init.zeros_(part[-1].bias)

    def encode(self, x):
        z = self.enc(F.pixel_unshuffle(x, 4))
        return z + self.er(z)

    def decode(self, z):
        return F.pixel_shuffle(self.dec(z + self.dr(z)), 4)

    def forward(self, x):
        return self.decode(self.encode(x))


class Encoder(nn.Module):
    """Shared source encoder at full, half, and quarter resolution."""

    def __init__(self):
        super().__init__()
        self.full = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), Block(32), Block(32))
        self.half_stage = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1), Block(64), Block(64)
        )
        self.quarter = nn.Sequential(
            nn.Conv2d(64, 96, 3, 2, 1), Block(96), Block(96)
        )

    def forward(self, x):
        full = self.full(x)
        half = self.half_stage(full)
        quarter = self.quarter(half)
        return full, half, quarter


class GlobalPrior(nn.Module):
    """Encode the full source pair and coordinates as spatial context tokens."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.SiLU(), Block(32),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(), Block(64),
            nn.Conv2d(64, 96, 3, 2, 1), Block(96),
        )

    def forward(self, whole):
        batch, _, height, width = whole.shape
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=whole.device),
            torch.linspace(-1, 1, width, device=whole.device), indexing="ij",
        )
        xy = torch.stack([x, y])[None].expand(batch, -1, -1, -1)
        return self.net(torch.cat([whole, xy], 1)).flatten(2).transpose(1, 2)


class Field(nn.Module):
    """Patchwise latent velocity with explicit context cross-attention."""

    def __init__(self, width):
        super().__init__()
        self.input = nn.Conv2d(16 * 3 + 96 * 2 + 2 + 2, width, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(6, width), nn.SiLU(), nn.Linear(width, width))
        self.high = nn.Sequential(Block(width), Block(width))
        self.down = nn.Conv2d(width, 2 * width, 3, 2, 1)
        self.low = nn.Sequential(Block(2 * width), Block(2 * width), Block(2 * width))
        self.norm = nn.GroupNorm(8, 2 * width)
        self.query = nn.Conv2d(2 * width, 128, 1)
        self.key = nn.Linear(96, 128)
        self.value = nn.Linear(96, 128)
        self.project = nn.Conv2d(128, 2 * width, 1)
        self.up = nn.Conv2d(2 * width, width, 3, padding=1)
        self.merge = nn.Sequential(Block(width), Block(width))
        self.out = nn.Conv2d(width, 16, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)
        self.logvar = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.logvar[-1].weight)
        nn.init.zeros_(self.logvar[-1].bias)

    def forward(self, packed, r, t, tokens, thumbnail):
        thumb = F.interpolate(
            thumbnail, size=packed.shape[-2:], mode="bilinear", align_corners=False
        )
        time = torch.stack([t, r, t - r, t * t, torch.sin(torch.pi * t), torch.cos(torch.pi * r)], 1)
        high = self.high(
            self.input(torch.cat([packed, thumb], 1)) + self.time(time)[:, :, None, None]
        )
        low = self.low(self.down(high))
        normalized = self.norm(low)
        batch, _, height, width = normalized.shape
        query = self.query(normalized).reshape(batch, 4, 32, height * width).transpose(-1, -2)
        key = self.key(tokens).reshape(batch, -1, 4, 32).permute(0, 2, 3, 1)
        value = self.value(tokens).reshape(batch, -1, 4, 32).permute(0, 2, 1, 3)
        # Explicit operations retain forward-mode AD support for the JVP objective.
        attention = ((query @ key) / math.sqrt(32)).softmax(-1) @ value
        low = low + self.project(attention.transpose(-1, -2).reshape(batch, 128, height, width))
        high = self.merge(high + self.up(up(low, high)))
        return self.out(high)


class FusionLevel(nn.Module):
    """Latent-conditioned feature combination for the provisional decoder."""

    def __init__(self, channels):
        super().__init__()
        self.state = nn.Conv2d(16, channels, 1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1), nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.refine = Block(channels)

    def forward(self, infrared, visible, z):
        state = up(self.state(z), infrared)
        weights = self.gate(torch.cat([infrared, visible, state], 1)).sigmoid()
        return self.refine(weights * infrared + (1 - weights) * visible + state)


class DetailRouter(nn.Module):
    """Select source details and predict a locally bounded signed correction."""

    def __init__(self, channels):
        super().__init__()
        self.state = nn.Conv2d(16, 16, 1)
        self.global_map = nn.Conv2d(96, 16, 1)
        self.net = nn.Sequential(
            nn.Conv2d(2 * channels + 34, 32, 1), nn.SiLU(), Block(32)
        )
        self.head = nn.Conv2d(32, 3, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, infrared, visible, z, global_map, detail_ir, detail_vis):
        state = up(self.state(z), infrared)
        context = up(self.global_map(global_map), infrared)
        features = self.net(torch.cat([infrared, visible, state, context, detail_ir, detail_vis], 1))
        prediction = self.head(features)
        weights = prediction[:, :2].softmax(1)
        amplitude = torch.maximum(detail_ir.abs(), detail_vis.abs())
        correction = 0.1 * prediction[:, 2:3].tanh() * amplitude
        return weights[:, :1] * detail_ir + weights[:, 1:2] * detail_vis + correction


class CoReFusion(nn.Module):
    """The D1 architecture used by the released CoReFusion checkpoint.

    ``cache`` encodes the source pair; ``one_step`` returns fused luminance and
    its predicted latent state. Training also uses ``velocity`` and ``codec``.
    """

    def __init__(self, width=96):
        super().__init__()
        self.codec = Codec()
        self.encoder = Encoder()
        self.global_prior = GlobalPrior()
        self.field = Field(width)
        self.levels = nn.ModuleList([FusionLevel(channels) for channels in (32, 64, 96)])
        self.up1 = nn.Conv2d(96, 64, 3, padding=1)
        self.up0 = nn.Conv2d(64, 32, 3, padding=1)
        self.decode1 = nn.Sequential(Block(64), Block(64))
        self.decode0 = nn.Sequential(Block(32), Block(32))
        self.head = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.routers = nn.ModuleList([DetailRouter(32), DetailRouter(64)])

    def cache(self, ir, vis, whole, xy):
        """Cache image-only computations before evaluating overlapping patches."""
        return dict(
            zi=self.codec.encode(ir), zv=self.codec.encode(vis),
            fi=self.encoder(ir), fv=self.encoder(vis),
            tokens=self.global_prior(whole), whole=whole, xy=xy,
            pi=pyramid(ir)[1], pv=pyramid(vis)[1],
            global_hw=(whole.shape[-2] // 8, whole.shape[-1] // 8),
        )

    @staticmethod
    def geometry(t, sizes, strides):
        value = float(t.detach().mean())
        index = 0 if value > 2 / 3 else (1 if value > 1 / 3 else 2)
        return sizes[index], strides[index]

    def velocity(self, z, r, t, cache, p, stride, patch_batch):
        packed = torch.cat(
            [z, cache["zi"], cache["zv"], cache["fi"][2], cache["fv"][2], cache["xy"]], 1
        )
        batch, channels, height, width = packed.shape
        padded_h = max(p, math.ceil(max(0, height - p) / stride) * stride + p)
        padded_w = max(p, math.ceil(max(0, width - p) / stride) * stride + p)
        packed = F.pad(packed, (0, padded_w - width, 0, padded_h - height), mode="replicate")
        patches = F.unfold(packed, p, stride=stride).transpose(1, 2)
        count = patches.shape[1]
        patches = patches.reshape(batch * count, channels, p, p)
        ids = torch.arange(batch, device=z.device).repeat_interleave(count)
        outputs = []
        for start in range(0, len(patches), patch_batch):
            indices = ids[start:start + patch_batch]
            outputs.append(self.field(
                patches[start:start + patch_batch], r[indices], t[indices],
                cache["tokens"][indices], cache["whole"][indices],
            ))
        output = torch.cat(outputs).reshape(batch, count, 16 * p * p).transpose(1, 2)
        canvas = F.fold(output, (padded_h, padded_w), p, stride=stride)
        overlap = F.fold(z.new_ones(1, p * p, count), (padded_h, padded_w), p, stride=stride)
        return (canvas / overlap)[..., :height, :width]

    def provisional(self, z, cache):
        """Decode a full-resolution provisional image before coarse pooling."""
        features = [
            module(ir, vis, z)
            for module, ir, vis in zip(self.levels, cache["fi"], cache["fv"])
        ]
        half = self.decode1(features[1] + self.up1(up(features[2], features[1])))
        full = self.decode0(features[0] + self.up0(up(half, features[0])))
        return self.codec.decode(z) + 0.2 * torch.tanh(self.head(full))

    def decode(self, z, cache):
        coarse = F.avg_pool2d(self.provisional(z, cache), 4)
        height, width = cache["global_hw"]
        tokens = cache["tokens"].transpose(1, 2).reshape(len(z), 96, height, width)
        aligned = F.grid_sample(
            tokens, cache["xy"].permute(0, 2, 3, 1),
            mode="bilinear", padding_mode="border", align_corners=False,
        )
        details = [
            router(cache["fi"][scale], cache["fv"][scale], z, aligned,
                   cache["pi"][scale], cache["pv"][scale])
            for scale, router in enumerate(self.routers)
        ]
        return reconstruct(coarse, details)

    def one_step(self, cache, cfg, noise=None):
        z = torch.randn_like(cache["zi"]) if noise is None else noise
        t = z.new_ones(len(z))
        r = z.new_zeros(len(z))
        patch, stride = self.geometry(t, cfg.PATCH_SIZES, cfg.PATCH_STRIDES)
        velocity = self.velocity(z, r, t, cache, patch, stride, cfg.PATCH_BATCH)
        terminal = z - velocity
        return self.decode(terminal, cache), terminal

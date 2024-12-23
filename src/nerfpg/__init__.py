from typing import Callable

from PIL import Image
import torch as tch
from torch import nn
import numpy as np


def image_to_tensor(img: Image, dev: tch.device) -> tch.Tensor:
    return (tch.tensor(np.array(img)).to(dev).type(tch.float32) / 255.0).permute(
        (2, 0, 1)
    )


def tensor_to_image(t: tch.Tensor) -> Image:
    s = (t.clip(0.0, 1.0) * 255.0).type(tch.uint8).permute((1, 2, 0))
    return Image.fromarray(s.numpy(force=True))


def grid(shape: tuple[int, int], dev: tch.device) -> tch.Tensor:
    """
    Rectangular grid of coordinates.
    Returned tensor shape: (shape[0], shape[1], 2)
    """
    x = tch.arange(shape[0]) / (shape[0] - 1)
    x = tch.stack((x, tch.ones_like(x)), dim=-1)
    x = x.unsqueeze(1).to(dev)

    y = tch.arange(shape[1]) / (shape[1] - 1)
    y = tch.stack((tch.ones_like(y), y), dim=-1)
    y = y.unsqueeze(0).to(dev)

    return x * y


def encode_coord(coord: tch.Tensor, n_modes: int, factor: float = 1.0) -> tch.Tensor:
    """
    Cosine coordinate encoding.
    + `coord` shape: (N, C)
    + Returned tensor shape: (N, C * n_modes)
    """
    freqs = tch.arange(1, n_modes + 1, device=coord.device) * np.pi * factor

    args = coord.unsqueeze(-1) * freqs.reshape((1, 1, -1))
    return args.cos().flatten(1, 2)


class Lambda(nn.Module):
    def __init__(self, func: Callable[[tch.Tensor], tch.Tensor]):
        super().__init__()
        self.func = func

    def forward(self, x: tch.Tensor) -> tch.Tensor:
        return self.func(x)

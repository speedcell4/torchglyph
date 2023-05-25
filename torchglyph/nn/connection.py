from typing import Type, Union

import torch
from torch import Tensor

from torchglyph.nn.normalization import LayerNorm


class PreLayerNorm(LayerNorm):
    def forward(self, tensor: Tensor, *, sub_layer, **kwargs):
        out = sub_layer(super(PreLayerNorm, self).forward(tensor), **kwargs)

        if torch.is_tensor(out):
            return tensor + out

        out, *args = out
        return tensor + out, *args


class PostLayerNorm(LayerNorm):
    def forward(self, tensor: Tensor, *, sub_layer, **kwargs):
        out = sub_layer(tensor, **kwargs)

        if torch.is_tensor(out):
            return super(PostLayerNorm, self).forward(tensor + out)

        out, *args = out
        return super(PostLayerNorm, self).forward(tensor + out), *args


Connections = Union[
    Type[PreLayerNorm],
    Type[PostLayerNorm],
]

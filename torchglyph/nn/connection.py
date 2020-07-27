from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence


class ResNorm(nn.Module):
    """
    https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    """

    def __init__(self, input_dim: int, *, sub_layer: nn.Module) -> None:
        super(ResNorm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.sub_layer = sub_layer
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        z = self.sub_layer(x, *args, **kwargs)
        if torch.is_tensor(z):
            return self.layer_norm(x + z)
        elif isinstance(z, PackedSequence):
            return z._replace(data=self.layer_norm(x.data + z.data))
        else:
            raise NotImplementedError


class DenseNorm(nn.Module):
    """
    http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf
    """

    def __init__(self, input_dim: int, *, sub_layer: nn.Module) -> None:
        super(DenseNorm, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim * 2

        self.sub_layer = sub_layer
        self.layer_norm = nn.LayerNorm(input_dim * 2)

    def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        z = self.sub_layer(x, *args, **kwargs)
        if torch.is_tensor(z):
            return self.layer_norm(torch.cat([x, z], dim=-1))
        elif isinstance(z, PackedSequence):
            return z._replace(data=self.layer_norm(torch.cat([x.data, z.data], dim=-1)))
        else:
            raise NotImplementedError


class ReZero(nn.Module):
    """
    https://arxiv.org/pdf/2003.04887.pdf
    """

    def __init__(self, input_dim: int, *, sub_layer: nn.Module) -> None:
        super(ReZero, self).__init__()
        self.input_dim = input_dim
        self.output_dim = input_dim

        self.sub_layer = sub_layer
        self.scale = nn.Parameter(
            torch.tensor([0.], dtype=torch.float32),
            requires_grad=True,
        )

    def extra_repr(self) -> str:
        return f'(scale): Parameter({self.scale.data})'

    def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        z = self.sub_layer(x, *args, **kwargs)
        if torch.is_tensor(z):
            return x + z * self.scale
        elif isinstance(z, PackedSequence):
            return z._replace(data=x.data + z.data * self.scale)
        else:
            raise NotImplementedError

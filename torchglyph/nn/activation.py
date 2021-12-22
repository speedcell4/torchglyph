from torch import nn, Tensor
from torch.nn import GLU, Tanh, ReLU, GELU, SELU

__all__ = [
    'GLU', 'GTU',
    'Tanh', 'ReLU', 'GELU', 'SELU',
]


class GTU(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(GTU, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def forward(self, tensor: Tensor) -> Tensor:
        a, b = tensor.chunk(2, dim=self.dim)
        return a.tanh_() * b.sigmoid_()

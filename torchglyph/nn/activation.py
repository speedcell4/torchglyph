from typing import Type, Union

from torch import nn


class Identity(nn.Identity):
    def __init__(self) -> None:
        super(Identity, self).__init__()


class ReLU(nn.ReLU):
    def __init__(self) -> None:
        super(ReLU, self).__init__()


class GELU(nn.GELU):
    def __init__(self) -> None:
        super(GELU, self).__init__()


class SiLU(nn.SiLU):
    def __init__(self) -> None:
        super(SiLU, self).__init__()


class ELU(nn.ELU):
    def __init__(self) -> None:
        super(ELU, self).__init__()


class Tanh(nn.Tanh):
    def __init__(self) -> None:
        super(Tanh, self).__init__()


Activations = Union[
    Type[Identity],
    Type[ReLU],
    Type[GELU],
    Type[SiLU],
    Type[ELU],
    Type[Tanh],
]

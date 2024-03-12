from typing import Type

from torchglyph.nn.plm.abc import *
from torchglyph.nn.plm.bart import *
from torchglyph.nn.plm.bert import *
from torchglyph.nn.plm.deberta import *
from torchglyph.nn.plm.gpt import *
from torchglyph.nn.plm.roberta import *
from torchglyph.nn.plm.xlnet import *


def full(model: nn.Module, /, **kwargs) -> None:
    model.requires_grad_(True)


def frozen(model: nn.Module, /, **kwargs) -> None:
    model.requires_grad_(False)


Tuning = Union[
    Type[full],
    Type[qof],
    Type[frozen],
]

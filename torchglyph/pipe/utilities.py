from typing import Tuple

from torch import Tensor

THRESHOLD = 8


def cum_tok(t: Tensor, a: int) -> Tuple[Tensor, int]:
    return t + a, a + t.size(0)


def cum_seq(t: Tensor, a: int) -> Tuple[Tensor, int]:
    return t + a, a + 1

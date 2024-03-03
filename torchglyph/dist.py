from logging import getLogger
from typing import Any, List

import torch
from torch import Generator, Tensor, distributed

logger = getLogger(__name__)


def get_rank() -> int:
    if not distributed.is_initialized():
        return 0

    return distributed.get_rank()


def is_master() -> bool:
    return get_rank() == 0


def get_world_size() -> int:
    if not distributed.is_initialized():
        return 1

    return distributed.get_world_size()


def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device('cpu')

    return torch.device(f'cuda:{get_rank()}')


def get_generator() -> Generator:
    if not torch.cuda.is_available():
        return torch.default_generator

    return torch.cuda.default_generators[distributed.get_rank()]


def all_reduce(tensor: Tensor) -> Tensor:
    if not distributed.is_initialized():
        return tensor

    return distributed.all_reduce(tensor)


def all_gather_object(obj: Any, word_size: int = None) -> List[Any]:
    if not distributed.is_initialized():
        return [obj]

    if word_size is None:
        word_size = distributed.get_world_size()

    gather_list = [None for _ in range(word_size)]
    distributed.all_gather_object(gather_list, obj)
    return gather_list


def all_gather(tensor: Tensor, word_size: int = None) -> List[Tensor]:
    if not distributed.is_initialized():
        return [tensor]

    if word_size is None:
        word_size = distributed.get_world_size()

    gather_list = [
        torch.empty(size, dtype=dtype, device=tensor.device)
        for size, dtype in zip(
            all_gather_object(tensor.size(), word_size=word_size),
            all_gather_object(tensor.dtype, word_size=word_size),
        )
    ]
    distributed.all_gather(gather_list, tensor)
    return gather_list

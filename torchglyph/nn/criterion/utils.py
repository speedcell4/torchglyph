from torch import distributed

from torchglyph.env import all_gather_object


def get_scaling(batch_size: int) -> float:
    if distributed.is_initialized():
        batch_sizes = all_gather_object(batch_size)
        return sum(batch_sizes) / (len(batch_sizes) * batch_size)

    return 1.0

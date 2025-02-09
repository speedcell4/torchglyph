from accelerate.utils import gather_object
from torch import distributed


def get_scaling(batch_size: int) -> float:
    if distributed.is_initialized():
        batch_sizes = gather_object([batch_size])
        return sum(batch_sizes) / (len(batch_sizes) * batch_size)

    return 1.0

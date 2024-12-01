from pathlib import Path
from typing import Any, List

from torch import Tensor, distributed

from torchglyph.serde import load_args, load_sota, save_args, save_sota


def all_gather_object(obj: Any) -> List[Any]:
    object_list = [None for _ in range(distributed.get_world_size())]
    distributed.all_gather_object(obj=obj, object_list=object_list)
    return object_list


def all_gather(tensor: Tensor) -> List[Tensor]:
    tensor_list = [tensor.new_empty(size=size) for size in all_gather_object(tensor.size())]
    distributed.all_gather(tensor=tensor, tensor_list=tensor_list)
    return tensor_list


def link_checkpoint(out_dir: Path, checkpoint: Path, prefix: str = 'co') -> None:
    if distributed.is_initialized() and distributed.get_rank() != 0:
        return

    if checkpoint.is_file():
        checkpoint = checkpoint.parent

    save_args(
        out_dir=out_dir, **{
            f'{prefix}-{key}': value
            for key, value in load_args(out_dir=checkpoint).items()
        }
    )

    save_sota(
        out_dir=out_dir, **{
            f'{prefix}-{key}': value
            for key, value in load_sota(out_dir=checkpoint).items()
        }
    )

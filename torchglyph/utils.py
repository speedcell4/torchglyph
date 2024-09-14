from pathlib import Path

from torch import distributed

from torchglyph.serde import load_args, load_sota, save_args, save_sota


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

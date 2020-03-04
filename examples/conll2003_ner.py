from pathlib import Path
from typing import List, Tuple, Dict

from torchglyph.dataset import Dataset, Pipeline, DataLoader
from torchglyph.io import conllx_iter
from torchglyph.pipelines import SeqPipe


class CoNLL2003(Dataset):
    def __init__(self, path: Path, pipelines: List[Dict[str, Pipeline]]) -> None:
        super(CoNLL2003, self).__init__(
            instances=[
                [datum for datum in zip(*sentence)]
                for sentence in conllx_iter(path)
            ],
            pipelines=pipelines,
        )

    @classmethod
    def iters(cls, *paths: Path, batch_size: int) -> Tuple[DataLoader, ...]:
        word = SeqPipe()
        xpos = SeqPipe()
        target = SeqPipe()

        train, dev, test = tuple(cls(path, pipelines=[
            {'word': word},
            {'xpos': xpos},
            {},
            {},
            {'target': target},
        ]) for path in paths)

        word.build_vocab(train, dev, test)
        xpos.build_vocab(train)
        target.build_vocab(train)

        return DataLoader.iters(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )

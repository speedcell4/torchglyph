import logging
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Tuple, Callable, List

import torch
from torch import Tensor
from torch.nn import init
from tqdm import tqdm

from torchglyph import data_path
from torchglyph.io import download_and_unzip


class Vocab(object):
    def __init__(self, counter: Counter,
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 max_size: Optional[int] = None, min_freq: int = 1) -> None:
        super(Vocab, self).__init__()

        self.counter = counter
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens
        self.unk_idx = 0
        self.max_size = max_size
        self.min_freq = min_freq

        self.stoi = {}
        self.itos = []
        self.vectors: Optional[Tensor] = None

        if unk_token is not None:
            self.unk_idx = self.add_token_(unk_token)
            self.stoi = defaultdict(self._default_factory, **self.stoi)

        for token in (pad_token, *special_tokens):
            if token is not None:
                self.add_token_(token)

        for token, freq in counter.most_common(n=max_size):
            if freq < min_freq:
                break
            self.add_token_(token)

    def _default_factory(self) -> int:
        return self.unk_idx

    def add_token_(self, token) -> int:
        assert token is not None

        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

        return self.stoi[token]

    def __repr__(self) -> str:
        args = ', '.join([a for a in [
            f'tok={self.__len__()}',
            f'dim={self.vectors.size(1)}' if self.vectors is not None else None,
            f'unk_token={self.unk_token}',
        ] if a is not None])
        return f'{self.__class__.__name__}({args})'

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def __and__(self, rhs: Union['Counter', 'Vocab']) -> 'Vocab':
        if isinstance(rhs, Vocab):
            rhs = rhs.counter
        counter = Counter({
            token: freq
            for token, freq in self.counter.items()
            if token in rhs
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size, min_freq=self.min_freq,
        )

    def __add__(self, rhs: Union['Counter', 'Vocab']) -> 'Vocab':
        if isinstance(rhs, Vocab):
            rhs = rhs.counter
        counter = Counter({
            token: self.counter[token] + rhs[token]
            for token in {*self.counter.keys(), *rhs.keys()}
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size, min_freq=self.min_freq,
        )

    def __sub__(self, rhs: Union['Counter', 'Vocab']) -> 'Vocab':
        if isinstance(rhs, Vocab):
            rhs = rhs.counter
        counter = Counter({
            token: freq
            for token, freq in self.counter.items()
            if token not in rhs
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size, min_freq=self.min_freq,
        )

    def load_vectors(self, vectors: 'Vectors') -> None:
        self.vectors = torch.empty((len(self), vectors.vec_dim), dtype=torch.float32)
        for token, index in self.stoi.items():
            vectors.update_(token, self.vectors[index])

        if self.pad_token is not None:
            init.zeros_(self.vectors[self.stoi[self.pad_token]])


class Vectors(Vocab):
    def __init__(self, urls_dest: List[Tuple[str, Path]], path: Path,
                 unk_init_: Callable[[Tensor], Tensor] = init.normal_) -> None:
        super(Vectors, self).__init__(
            counter=Counter(),
            unk_token=None, pad_token=None,
            special_tokens=(), max_size=None, min_freq=1,
        )

        self.vectors = []
        self.unk_init_ = unk_init_

        pt_path = path.with_suffix('.pt')
        if not pt_path.exists():
            if not path.exists():
                for url, dest in urls_dest:
                    download_and_unzip(url, dest)

            with path.open('rb') as fp:
                vec_dim = None

                for raw in tqdm(fp, desc=f'reading {path}', unit=' tokens'):  # type:bytes
                    token, *vs = raw.rstrip().split(b' ')

                    if vec_dim is None:
                        vec_dim = len(vs)
                    elif vec_dim != len(vs):
                        raise ValueError(f'vector dimensions are not consistent, {vec_dim} != {len(vs)}')

                    self.add_token_(str(token, encoding='utf-8'))
                    self.vectors.append(torch.tensor([float(v) for v in vs], dtype=torch.float32))

            self.vectors = torch.stack(self.vectors, 0)
            logging.info(f'saving vectors to {pt_path}')
            torch.save((self.itos, self.stoi, self.vectors), pt_path)
        else:
            logging.info(f'loading vectors from {pt_path}')
            self.itos, self.stoi, self.vectors = torch.load(pt_path)

    @torch.no_grad()
    def update_(self, token: str, vector: Tensor) -> None:
        if token in self:
            vector[:] = self.vectors[self.stoi[token]]
        else:
            self.unk_init_(vector)


class Glove(Vectors):
    def __init__(self, name: str, dim: int) -> None:
        super(Glove, self).__init__(
            urls_dest=[
                (f'http://nlp.stanford.edu/data/glove.{name}.zip', data_path / f'glove.{name}' / f'glove.{name}.zip'),
            ],
            path=data_path / f'glove.{name}' / f'glove.{name}.{dim}d.txt',
        )


if __name__ == '__main__':
    vectors = Glove('6B', 50)
    print(f'vectors => {vectors}')
    vectors.vectors = None
    print(f'vectors => {vectors}')

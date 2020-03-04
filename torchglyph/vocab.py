import logging
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Union, Optional, Tuple

import torch
from tqdm import tqdm


class Vocab(object):
    def __init__(self, counter: Counter,
                 unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 max_size: Optional[int] = None, min_freq: int = 1) -> None:
        super(Vocab, self).__init__()

        self.counter = counter
        self.unk_token = unk_token
        self.special_tokens = special_tokens
        self.unk_idx = 0
        self.max_size = max_size
        self.min_freq = min_freq

        self.stoi = {}
        self.itos = []

        if unk_token is not None:
            self.unk_idx = self.add_token_(unk_token)
            self.stoi = defaultdict(self._default_factory, **self.stoi)

        for token in special_tokens:
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
        args = ', '.join([
            f'size={self.__len__()}',
            f'unk_token={self.unk_token}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __len__(self) -> int:
        return len(self.stoi)

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
            unk_token=self.unk_token, special_tokens=self.special_tokens,
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
            unk_token=self.unk_token, special_tokens=self.special_tokens,
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
            unk_token=self.unk_token, special_tokens=self.special_tokens,
            max_size=self.max_size, min_freq=self.min_freq,
        )


class Vectors(object):
    def __init__(self, path: Path) -> None:
        vocab = Vocab(Counter(), unk_token=None, special_tokens=())
        vectors = []

        pt_path = path.with_suffix('.ptptpt')
        if not pt_path.exists():
            with path.open('rb') as fp:
                token_dim = None
                for raw in tqdm(fp, desc=str(path)):  # type:bytes
                    token, *vs = raw.rstrip().split(b' ')

                    if token_dim is None:
                        token_dim = len(vs)
                    elif token_dim != len(vs):
                        raise ValueError(f'vector dimensions are not consistent, {token_dim} != {len(vs)}')

                    vocab.add_token_(str(token, encoding='utf-8'))
                    vectors.append(torch.tensor([float(v) for v in vs], dtype=torch.float32))

            vectors = torch.stack(vectors, 0)
            logging.info(f'saving vectors to {pt_path}')
            torch.save((vocab.itos, vocab.stoi, vectors, token_dim), pt_path)
        else:
            logging.info(f'loading vectors from {pt_path}')
            vocab.itos, vocab.stoi, vectors, token_dim = torch.load(pt_path)

        self.vocab = vocab
        self.vectors = vectors
        self.token_dim = token_dim

from collections import Counter, defaultdict
from typing import Optional, Tuple


class Vocab(object):
    def __init__(self, counter: Counter,
                 unk_token: Optional[str] = '<unk>',
                 special_tokens: Tuple[Optional[str], ...] = (),
                 max_size: Optional[int] = None, min_freq: int = 1) -> None:
        super(Vocab, self).__init__()

        self.counter = counter
        self.unk_token = unk_token
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

        for token, freq in counter.most_common(n=max_size - len(self.stoi)):
            if freq < min_freq:
                break
            self.add_token_(token)

    def _default_factory(self) -> int:
        return self.unk_idx

    def add_token_(self, token) -> int:
        assert token is not None

        idx = len(self.itos)
        self.stoi[token] = idx
        self.itos.append(token)
        return idx

    def __repr__(self) -> str:
        args = ', '.join([
            f'size={self.__len__()}',
            f'unk_token={self.unk_token}',
            f'max_size={self.max_size}',
            f'min_freq={self.min_freq}',
        ])
        return f'{self.__class__.__name__}({args})'

    def __len__(self) -> int:
        return len(self.stoi)

    def __and__(self, other: 'Vocab') -> 'Vocab':
        counter = Counter({
            token: freq
            for token, freq in self.counter.items()
            if token in other.counter
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )

    def __add__(self, other: 'Vocab') -> 'Vocab':
        counter = Counter({
            token: self.counter[token] + other.counter[token]
            for token in {*self.counter.keys(), *other.counter.keys()}
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )

    def __sub__(self, other: 'Vocab') -> 'Vocab':
        counter = Counter({
            token: freq
            for token, freq in self.counter.items()
            if token not in other.counter
        })
        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )

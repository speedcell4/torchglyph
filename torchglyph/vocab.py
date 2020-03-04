from collections import Counter, defaultdict
from typing import Optional, Tuple, Union


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

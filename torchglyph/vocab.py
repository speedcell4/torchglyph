import logging
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch import Tensor
from torch.nn import init
from tqdm import tqdm

from torchglyph import data_dir
from torchglyph.io import DownloadMixin

logger = logging.getLogger(__name__)

__all__ = [
    'Vocab', 'Vectors',
    'Glove', 'FastText',
]


class Vocab(object):
    def __init__(self, counter: Counter,
                 unk_token: Optional[str],
                 pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 max_size: Optional[int] = None, min_freq: int = 1) -> None:
        super(Vocab, self).__init__()

        if max_size is not None:
            counter = Counter(counter.most_common(n=max_size))

        self.freq = counter
        self.max_size = max_size
        self.min_freq = min_freq

        self.itos = []
        self.stoi = defaultdict(self._default_factory)
        self.vectors: Optional[Tensor] = None

        self.unk_idx = None
        if unk_token is not None:
            self.unk_idx = self.add_token_(unk_token)

        self.pad_idx = None
        if pad_token is not None:
            self.pad_idx = self.add_token_(pad_token)

        for token in special_tokens:
            self.add_token_(token)

        self.unk_token = unk_token
        self.pad_token = pad_token

        special_tokens = (unk_token, pad_token, *special_tokens)
        self.special_tokens = tuple(token for token in special_tokens if token is not None)

        for token, freq in self.freq.most_common():
            if freq < min_freq:
                break
            self.add_token_(token)

    def _default_factory(self) -> Optional[int]:
        return self.unk_idx

    def add_token_(self, token) -> int:
        assert token is not None

        if token not in self.stoi:
            self.stoi[token] = len(self.stoi)
            self.itos.append(token)

        return self.stoi[token]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'{len(self.stoi)}',
            *self.special_tokens,
        ])

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def union(self, other: 'Vocab', *fallbacks) -> 'Vocab':
        counter = Counter()

        for token, freq in self.freq.items():
            if token in other.stoi:
                counter[token] = freq
            else:
                for fallback in fallbacks:
                    fallback_token = fallback(token)
                    if fallback_token in other.stoi:
                        counter[fallback_token] = freq
                        break

        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )

    @torch.no_grad()
    def load_vectors(self, *fallbacks, vectors: 'Vectors') -> Tuple[int, int]:
        self.vectors = torch.empty((len(self), vectors.vectors.size()[1]), dtype=torch.float32)

        tok, occ = 0, 0
        for token, index in self.stoi.items():
            if vectors.query_(token, self.vectors[index], *fallbacks):
                tok += 1
                occ += self.freq[token]

        if self.pad_token is not None:
            init.zeros_(self.vectors[self.stoi[self.pad_token]])

        return tok, occ

    def save(self, path: Path) -> None:
        logger.info(f'saving {self.__class__.__name__} to {path}')
        torch.save((self.stoi, self.itos, self.vectors), path)

    def load(self, path: Path) -> None:
        logger.info(f'loading {self.__class__.__name__} from {path}')
        self.stoi, self.itos, self.vectors = torch.load(path)


class Vectors(Vocab, DownloadMixin):
    vector_format: str

    def __init__(self, root: Path = data_dir, **kwargs) -> None:
        super(Vectors, self).__init__(
            counter=Counter(),
            unk_token=None,
            pad_token=None,
            special_tokens=(),
            max_size=None, min_freq=1,
        )

        path, = self.paths(root=root, **kwargs)  # type:Path
        self.cache_(path=path)

    def cache_(self, path: Path) -> None:
        torch_path = path.with_suffix('.pt')

        if not torch_path.exists():
            vectors = []
            with path.open('r', encoding='utf-8', errors='replace') as fp:

                num_vectors, vector_size = None, None
                if self.vector_format == 'word2vec':
                    num_vectors, vector_size = map(int, next(fp).strip().split(' '))

                for raw in tqdm(fp, desc=f'caching {path}', unit=' tokens', total=num_vectors):
                    token, *vector = raw.rstrip().split(' ')

                    if vector_size is None:
                        vector_size = len(vector)
                    assert vector_size == len(vector), f'{vector_size} != {len(vector)} :: {token}'

                    self.add_token_(token)
                    vectors.append([float(v) for v in vector])

            self.vectors = torch.tensor(vectors, dtype=torch.float32)
            self.save(torch_path)
        else:
            self.load(torch_path)

    @torch.no_grad()
    def query_(self, token: str, tensor: Tensor, *fallbacks) -> bool:
        if token in self:
            tensor[:] = self.vectors[self.stoi[token]]
            return True

        for fallbacks in fallbacks:
            new_token = fallbacks(token)
            if new_token in self:
                tensor[:] = self.vectors[self.stoi[new_token]]
                return True

        self.unk_init_(tensor)
        return False

    @staticmethod
    def unk_init_(tensor: Tensor) -> None:
        init.normal_(tensor)


class Glove(Vectors):
    vector_format = 'glove'

    def __init__(self, name: str, dim: int, root: Path = data_dir) -> None:
        super(Glove, self).__init__(root=root, name=name, dim=dim)

    def get_urls(self, name: str, dim: int) -> List[Tuple[str, ...]]:
        return [(
            f'https://nlp.stanford.edu/data/glove.{name}.zip',
            f'glove.{name}.zip',
            f'glove.{name}.{dim}d.txt',
        )]


class FastText(Vectors):
    vector_format = 'word2vec'

    def __init__(self, name: str, lang: str, root: Path = data_dir) -> None:
        super(FastText, self).__init__(root=root, name=name, lang=lang)

    def get_urls(self, name: str, lang: str) -> List[Tuple[str, ...]]:
        if name == 'cc':
            return [(
                f'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz',
                f'cc.{lang}.300.vec.gz',
                f'cc.{lang}.300.vec',
            )]
        if name == 'wiki':
            return [(
                f'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang}.vec',
                f'wiki.{lang}.vec',
                f'wiki.{lang}.vec',
            )]
        if name == 'aligned':
            return [(
                f'https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.{lang}.align.vec',
                f'wiki.{lang}.align.vec',
                f'wiki.{lang}.align.vec',
            )]

        raise KeyError(f'{name} is not supported')

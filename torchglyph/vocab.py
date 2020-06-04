import logging
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import torch
from torch import Tensor
from torch.nn import init
from tqdm import tqdm

from torchglyph import data_path
from torchglyph.io import download_and_unzip

logger = logging.getLogger(__name__)


class Vocab(object):
    def __init__(self, counter: Counter,
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 max_size: Optional[int] = None, min_freq: int = 1) -> None:
        super(Vocab, self).__init__()

        if max_size is not None:
            counter = Counter(counter.most_common(n=max_size))

        self.freq = counter
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

        for token, freq in self.freq.most_common(n=max_size):
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

    def extra_repr(self) -> str:
        return ', '.join([a for a in [
            f"tok={self.__len__()}",
            None if self.vectors is None else f"dim={self.vec_dim}",
            None if self.unk_token is None else f"unk_token='{self.unk_token}'",
            None if self.pad_token is None else f"pad_token='{self.pad_token}'",
        ] if a is not None])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def __len__(self) -> int:
        return len(self.stoi)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def union(self, rhs: 'Vocab', *fallback_fns) -> 'Vocab':
        counter = Counter()

        for token, freq in self.freq.items():
            if token in rhs.stoi:
                counter[token] = freq
            else:
                for fallback_fn in fallback_fns:
                    new_token = fallback_fn(token)
                    if new_token in rhs.stoi:
                        counter[new_token] = freq
                        break

        return Vocab(
            counter=counter,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )

    @property
    def pad_idx(self) -> Optional[int]:
        if self.pad_token is None:
            return None
        return self.stoi[self.pad_token]

    @property
    def vec_dim(self) -> int:
        if self.vectors is None:
            return 0
        return self.vectors.size(1)

    def load_vectors(self, *fallback_fns, vectors: 'Vectors') -> Tuple[int, int]:
        self.vectors = torch.empty((len(self), vectors.vec_dim), dtype=torch.float32)

        tok, occ = 0, 0
        for token, index in self.stoi.items():
            if vectors.query_(token, self.vectors[index], *fallback_fns):
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


class Vectors(Vocab):
    def __init__(self, urls_dest: List[Tuple[str, Path]], path: Path,
                 heading: bool, unicode_error: str = 'replace', dtype: torch.dtype = torch.float32,
                 unk_init_: Callable[[Tensor], Tensor] = init.normal_) -> None:
        super(Vectors, self).__init__(
            counter=Counter(),
            unk_token=None, pad_token=None,
            special_tokens=(), max_size=None, min_freq=1,
        )

        vectors = []
        self.unk_init_ = unk_init_

        dump_path = path.with_suffix('.pt')
        if not dump_path.exists():
            if not path.exists():
                for url, dest in urls_dest:
                    download_and_unzip(url, dest)

            with path.open('rb') as fp:
                vector_dim = None

                for raw in tqdm(fp, desc=f'reading {path}', unit=' lines'):  # type: bytes
                    if heading:
                        _, vector_dim = map(int, raw.rstrip().split(b' '))
                        heading = False
                        continue
                    token, *vs = raw.rstrip().split(b' ')

                    if vector_dim is None:
                        vector_dim = len(vs)
                    elif vector_dim != len(vs):
                        logger.error(f'vector dimensions are not consistent, {vector_dim} != {len(vs)} :: {token}')
                        continue

                    self.add_token_(str(token, encoding='utf-8', errors=unicode_error))
                    vectors.append(torch.tensor([float(v) for v in vs], dtype=dtype))

            self.vectors = torch.stack(vectors, 0)
            self.save(dump_path)
        else:
            self.load(dump_path)

    @torch.no_grad()
    def query_(self, token: str, vector: Tensor, *fallback_fns) -> bool:
        if token in self:
            vector[:] = self.vectors[self.stoi[token]]
            return True
        for fallback_fn in fallback_fns:
            new_token = fallback_fn(token)
            if new_token in self:
                vector[:] = self.vectors[self.stoi[new_token]]
                return True
        self.unk_init_(vector)
        return False


class Glove(Vectors):
    def __init__(self, name: str, dim: int) -> None:
        path = data_path / f'glove.{name}'
        super(Glove, self).__init__(
            urls_dest=[(
                f'http://nlp.stanford.edu/data/glove.{name}.zip',
                path / f'glove.{name}.zip'
            )],
            path=path / f'glove.{name}.{dim}d.txt', heading=False,
        )


class FastTest(Vectors):
    def __init__(self, lang: str) -> None:
        path = data_path / 'fasttext'
        super(FastTest, self).__init__(
            urls_dest=[(
                f'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.{lang}.vec',
                path / f'wiki.{lang}.vec',
            )],
            path=path / f'wiki.{lang}.vec', heading=True,
        )


class NLPLVectors(Vectors):
    def __init__(self, index: int, repository: str = '20', name: str = 'model.txt', heading: bool = False) -> None:
        path = data_path / 'nlpl' / f'{index}'
        super(NLPLVectors, self).__init__(
            urls_dest=[(
                f'http://vectors.nlpl.eu/repository/{repository}/{index}.zip',
                path / f'{index}.zip',
            )],
            path=path / name, heading=heading,
        )

import logging
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn

from torchglyph import data_dir
from torchglyph.formats.vector import load_word2vec, load_glove
from torchglyph.io import DownloadMixin

logger = logging.getLogger(__name__)

__all__ = [
    'Vocab', 'PreTrainedEmbedding',
    'Glove', 'FastText',
]


class Vocab(object):
    def __init__(self, unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 special_tokens: Tuple[str, ...] = ()) -> None:
        super(Vocab, self).__init__()

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self.special_tokens = tuple(
            token for token in {unk_token, pad_token, bos_token, eos_token, *special_tokens}
            if token is not None
        )

        self.index2token = {}
        self.token2index = {}

    @property
    def unk_index(self) -> int:
        return self.token2index.get(self.unk_token, None)

    @property
    def pad_index(self) -> int:
        return self.token2index.get(self.pad_token, None)

    @property
    def bos_index(self) -> int:
        return self.token2index.get(self.bos_token, None)

    @property
    def eos_index(self) -> int:
        return self.token2index.get(self.eos_token, None)

    def add_token(self, token: str) -> int:
        assert token is not None

        if token not in self.token2index:
            index = len(self.token2index)
            self.token2index[token] = index
            self.index2token[index] = token

        return self.token2index[token]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'{len(self)}',
            *self.special_tokens,
        ])

    def __len__(self) -> int:
        return len(self.token2index)

    def __contains__(self, token: str) -> bool:
        return token in self.token2index

    def __getitem__(self, token: str) -> int:
        return self.token2index.get(token, self.unk_index)

    def inv(self, sequence):
        if isinstance(sequence, int):
            return self.index2token.get(sequence, self.unk_token)
        else:
            return type(sequence)([self.inv(seq) for seq in sequence])

    def build(self, counter: Counter, max_size: int = None, min_freq: int = 0, weight: Tensor = None) -> 'Vocab':
        for token in self.special_tokens:
            self.add_token(token)

        for token, freq in counter.most_common(n=max_size):
            if freq < min_freq:
                break
            self.add_token(token)

        assert len(self.token2index) == len(self.index2token)

        self.counter = counter
        self.weight = weight

        logger.critical(f'counter and weight are updated by building vocabulary :: {self}')
        return self

    @torch.no_grad()
    def load_weight(self, *transforms, embedding: 'PreTrainedEmbedding') -> 'Vocab':
        self.weight = nn.Embedding(
            num_embeddings=len(self),
            embedding_dim=embedding.weight.size()[1],
            padding_idx=self.pad_index,
        ).weight.data.detach()
        logger.critical(f'weight is updated by loading {embedding}')

        matching = {}
        for token, index in self.token2index.items():
            if token in embedding.token2index:
                matching[token] = (index, embedding.token2index[token])
            else:
                for fn in transforms:
                    if fn(token) in embedding.token2index:
                        matching[token] = (index, embedding.token2index[fn(token)])
                        break

        occ_count, tok_count = 0, 0
        for token, (index1, index2) in matching.items():
            self.weight[index1] = embedding.weight[index2]
            occ_count += self.counter[token]
            tok_count += 1

        tok_ratio = tok_count / len(self)
        occ_ratio = occ_count / sum(self.counter.values())
        logger.info(f'{tok_ratio * 100:.2f}% tokens and {occ_ratio * 100:.2f}% occurrences are hit in {embedding}')
        return self


class PreTrainedEmbedding(Vocab, DownloadMixin):
    format: str

    def __init__(self, root: Path = data_dir, **kwargs) -> None:
        super(PreTrainedEmbedding, self).__init__()

        path, = self.paths(root=root, **kwargs)
        tokens, weight = self.load_cache(path=path)
        self.build(counter=Counter(tokens), max_size=None, min_freq=1, weight=weight)

    def load_cache(self, path: Path):
        cache = path.with_name(f'{path.name}.pt')
        cache.parent.mkdir(parents=True, exist_ok=True)

        if cache.exists():
            logger.info(f'loading from {cache}')
            tokens, weight = torch.load(cache, map_location=torch.device('cpu'))
        else:
            with path.open('r', encoding='utf-8') as fp:
                if self.format == 'glove':
                    tokens, weight = load_glove(fp=fp)
                elif self.format == 'word2vec':
                    tokens, weight = load_word2vec(fp=fp)
                else:
                    raise KeyError(f'{self.format} is not supported')

            logger.info(f'saving to {cache}')
            torch.save(obj=(tokens, weight), f=cache)

        return tokens, weight


class Glove(PreTrainedEmbedding):
    format = 'glove'

    def __init__(self, name: str, dim: int, root: Path = data_dir) -> None:
        super(Glove, self).__init__(root=root, name=name, dim=dim)

    @classmethod
    def get_urls(cls, name: str, dim: int) -> List[Tuple[str, ...]]:
        return [(
            f'https://nlp.stanford.edu/data/glove.{name}.zip',
            f'glove.{name}.zip',
            f'glove.{name}.{dim}d.txt',
        )]


class FastText(PreTrainedEmbedding):
    format = 'word2vec'

    def __init__(self, name: str, lang: str, root: Path = data_dir) -> None:
        super(FastText, self).__init__(root=root, name=name, lang=lang)

    @classmethod
    def get_urls(cls, name: str, lang: str) -> List[Tuple[str, ...]]:
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


if __name__ == '__main__':
    vectors = FastText(name='cc', lang='en')

import logging
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import Tensor
from tqdm import tqdm

from torchglyph.io import DownloadMixin, lock_folder

logger = getLogger(__name__)


class PreTrainedEmbedding(DownloadMixin):
    header: bool

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError

    @classmethod
    def load(cls, path: Path) -> Tuple[Tokenizer, Tensor]:
        tokenizer_path = path.parent.resolve() / f'{path.name}.json'
        embeddings_path = path.parent.resolve() / f'{path.name}.pt'

        with lock_folder(path=path.parent):
            if not tokenizer_path.exists() or not embeddings_path.exists():
                with path.open(mode='r', encoding='utf-8', newline='\n', errors='ignore') as fp:
                    if cls.header:
                        _, _ = fp.readline()

                    vocab, embeddings = {}, []
                    for index, raw in tqdm(enumerate(fp), desc=f'loading {path.name}'):
                        token, *embedding = raw.rstrip().split(' ')

                        vocab[token] = index
                        embeddings.append([float(e) for e in embedding])

                    tokenizer = Tokenizer(model=WordLevel(vocab=vocab, unk_token=None))
                    embeddings = torch.tensor(embeddings, dtype=torch.float32)

                logging.info(f'saving to {path.parent}')
                tokenizer.save(str(tokenizer_path))
                torch.save(embeddings, f=embeddings_path)
                return tokenizer, embeddings

        logging.info(f'loading from {path.parent}')
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        embeddings = torch.load(embeddings_path)
        return tokenizer, embeddings

    @classmethod
    def new(cls, **kwargs) -> Tuple[Tokenizer, Tensor]:
        path, = cls.paths(**kwargs)
        return cls.load(path=path)


class Glove6B(PreTrainedEmbedding):
    header = False

    @classmethod
    def urls(cls, dim: int, **kwargs) -> List[Tuple[str, ...]]:
        return [
            ('https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip', f'glove.6B.{dim}d.txt'),
        ]

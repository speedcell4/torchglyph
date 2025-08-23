from typing import IO, Tuple

import torch
from tqdm import tqdm


def loads_meta(string: str, *, sep: str = ' ') -> Tuple[int, int]:
    num_embeddings, embedding_dim = string.strip().split(sep=sep)
    return int(num_embeddings), int(embedding_dim)


def load_meta(fp: IO, *, sep: str = ' ') -> Tuple[int, int]:
    return loads_meta(fp.readline(), sep=sep)


def loads_vector(string: str, *, sep: str = ' '):
    token, *values = string.rstrip().split(sep=sep)
    return str(token), tuple(float(value) for value in values)


def load_vector(fp: IO, *, sep: str = ' '):
    return loads_vector(fp.readline(), sep=sep)


def load_word2vec(fp: IO, *, sep: str = ' '):
    num_embeddings, embedding_dim = load_meta(fp, sep=sep)

    tokens, vectors = [], []
    for string in tqdm(fp.readlines(), initial=0, unit=' tokens'):
        token, vector = loads_vector(string=string, sep=sep)
        assert embedding_dim == len(vector), f'{token} :: {embedding_dim} != {len(vector)}'

        tokens.append(token)
        vectors.append(vector)

    assert num_embeddings == len(tokens), f'{num_embeddings} != {len(tokens)}'
    return tokens, torch.tensor(vectors, dtype=torch.float32, device=torch.device('cpu'))


def load_glove(fp: IO, *, sep: str = ' '):
    token, vector = load_vector(fp, sep=sep)
    embedding_dim = len(vector)

    tokens, vectors = [token], [vector]
    for string in tqdm(fp.readlines(), initial=1, unit=' tokens'):
        token, vector = loads_vector(string, sep=sep)
        assert embedding_dim == len(vector), f'{token} :: {embedding_dim} != {len(vector)}'

        tokens.append(token)
        vectors.append(vector)

    return tokens, torch.tensor(vectors, dtype=torch.float32, device=torch.device('cpu'))

from typing import Iterable, List, Tuple, IO

from tqdm import tqdm

__all__ = [
    'load_meta',
    'loads_vector',
    'load_vector',
    'iter_vector',
    'load_word2vec', 'load_glove',
]

Token = str
Vector = Tuple[float, ...]


def load_meta(fp: IO, *, sep: str = ' ') -> Tuple[int, int]:
    num_embeddings, embedding_dim = next(fp).strip().split(sep=sep)
    return int(num_embeddings), int(embedding_dim)


def loads_vector(s: str, *, sep: str = ' ') -> Tuple[Token, Vector]:
    token, *scalars = s.strip().split(sep=sep)
    return str(token), tuple(float(scalar) for scalar in scalars)


def load_vector(fp: IO, *, sep: str = ' ') -> Tuple[List[Token], List[Vector]]:
    tokens, vectors = zip(*[loads_vector(s, sep=sep) for s in fp])
    return tokens, vectors


def iter_vector(fp: IO, *, sep: str = ' ') -> Iterable[Tuple[Token, Vector]]:
    yield from map(lambda s: loads_vector(s, sep=sep), fp)


def load_word2vec(fp: IO, *, sep: str = ' ') -> Tuple[List[Token], List[Vector]]:
    num_embeddings, embedding_dim = load_meta(fp, sep=sep)
    if isinstance(fp, tqdm) and hasattr(fp, 'total'):
        fp.total = num_embeddings

    tokens, vectors = [], []
    for token, vector in iter_vector(fp, sep=sep):
        assert len(vector) == embedding_dim, f'len({token}) = {len(vector)} != {embedding_dim}'

        tokens.append(token)
        vectors.append(vector)

    return tokens, vectors


def load_glove(fp: IO, *, sep: str = ' ') -> Tuple[List[Token], List[Vector]]:
    token, vector = loads_vector(next(fp), sep=sep)
    embedding_dim = len(vector)

    tokens, vectors = [token], [vector]
    for token, vector in iter_vector(fp, sep=sep):
        assert len(vector) == embedding_dim, f'len({token}) = {len(vector)} != {embedding_dim}'

        tokens.append(token)
        vectors.append(vector)

    return tokens, vectors

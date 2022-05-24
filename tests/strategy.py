import torch
from hypothesis import strategies as st

TINY_BATCH_SIZE = 5
TINY_TOKEN_SIZE = 11
TINY_EMBEDDING_DIM = 13
NUM_CONJUGATES = 5
NUM_TAGS = 7

if torch.cuda.is_available():
    BATCH_SIZE = 53
    TOKEN_SIZE = 83
    EMBEDDING_DIM = 107
    NUM_CONJUGATES = 5
    NUM_TAGS = 17
else:
    BATCH_SIZE = 37
    TOKEN_SIZE = 53
    EMBEDDING_DIM = 61
    NUM_CONJUGATES = 5
    NUM_TAGS = 17

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

torch.empty((1,), device=device)


@st.composite
def sizes(draw, *size: int, min_size: int = 1):
    max_size, *size = size
    n = draw(st.integers(min_value=min_size, max_value=max_size))

    if len(size) == 0:
        return n
    else:
        return draw(st.lists(sizes(*size), min_size=n, max_size=n))

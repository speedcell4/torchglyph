import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
torch.empty((1,), device=device)

if torch.cuda.is_available():
    MAX_BATCH_SIZE = 120
    TINY_BATCH_SIZE = 24

    MAX_TOKEN_SIZE = 512
    MAX_EMBEDDING_DIM = 100

else:
    MAX_BATCH_SIZE = 24
    TINY_BATCH_SIZE = 24

    MAX_TOKEN_SIZE = 128
    MAX_EMBEDDING_DIM = 50


def gen_sizes(*size: int, min_size: int = 1):
    max_size, *size = size
    n = torch.randint(low=min_size, high=max_size, size=()).item()

    if len(size) == 0:
        return n
    else:
        return [gen_sizes(*size, min_size=min_size) for _ in range(n)]

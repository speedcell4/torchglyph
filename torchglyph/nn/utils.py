import torch
from torch import Tensor

from torchglyph.vocab import Vocab


def partition_by_entropy(num_blocks: int, vocab: Vocab) -> Tensor:
    freq = torch.tensor([f for _, f in vocab.freq.most_common()], dtype=torch.float32)
    freq = freq / freq.sum(dim=-1, keepdim=True)
    entropy = (freq * freq.log()).neg().cumsum(dim=-1)
    threshold = torch.arange(num_blocks, dtype=torch.float32) * (entropy[-1] / num_blocks)

    return (entropy[None, :] >= threshold[:, None]).long().sum(dim=0) - 1

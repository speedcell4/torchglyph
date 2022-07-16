import torch
from torch import Tensor

from transformers import PreTrainedTokenizer


def uniform_token_mask(sequence: Tensor, tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    special_tokens = torch.tensor(tokenizer.all_special_ids, dtype=sequence.dtype, device=sequence.device)
    is_special = (sequence[..., None] == special_tokens).any(dim=-1)

    mask = torch.rand_like(sequence, dtype=torch.float32) < p
    mask[is_special] = False

    sequence = sequence.clone()
    sequence[mask] = tokenizer.mask_token_id

    return sequence


def uniform_span_mask(sequence: Tensor, duration: Tensor, tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    special_tokens = torch.tensor(tokenizer.all_special_ids, dtype=sequence.dtype, device=sequence.device)
    special_tokens = (sequence[..., None] == special_tokens).any(dim=-1)
    is_special = torch.segment_reduce(special_tokens.float(), reduce='max', lengths=duration).bool()

    mask = torch.rand_like(duration, dtype=torch.float32) < p
    mask[is_special] = False
    mask = torch.repeat_interleave(mask.float(), repeats=duration, dim=0).bool()

    sequence = sequence.clone()
    sequence[mask] = tokenizer.mask_token_id

    return sequence

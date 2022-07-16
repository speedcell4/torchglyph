from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Poisson
from transformers import PreTrainedTokenizer


def is_special(sequence: Tensor, duration: Optional[Tensor], tokenizer: PreTrainedTokenizer) -> Tensor:
    special_tokens = torch.tensor(tokenizer.all_special_ids, dtype=sequence.dtype, device=sequence.device)
    mask = (sequence[..., None] == special_tokens).any(dim=-1)
    if duration is not None:
        mask = torch.segment_reduce(mask.float(), reduce='max', lengths=duration).bool()
    return mask


def uniform_token_mask(sequence: Tensor, tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    mask = torch.rand_like(sequence, dtype=torch.float32) < p
    mask[is_special(sequence=sequence, duration=None, tokenizer=tokenizer)] = False

    sequence = sequence.clone()
    sequence[mask] = tokenizer.mask_token_id

    return sequence


def uniform_span_mask(sequence: Tensor, duration: Tensor, tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    mask = torch.rand_like(duration, dtype=torch.float32) < p
    mask[is_special(sequence=sequence, duration=duration, tokenizer=tokenizer)] = False
    mask = torch.repeat_interleave(mask.float(), repeats=duration, dim=0).bool()

    sequence = sequence.clone()
    sequence[mask] = tokenizer.mask_token_id

    return sequence


def poisson_span_mask(sequence: Tensor, duration: Tensor, tokenizer: PreTrainedTokenizer, rate: float) -> Tensor:
    mask = torch.rand_like(duration, dtype=torch.float32) < Poisson(rate=rate).log_prob(duration).exp()
    mask[is_special(sequence=sequence, duration=duration, tokenizer=tokenizer)] = False
    mask = torch.repeat_interleave(mask.float(), repeats=duration, dim=0).bool()

    sequence = sequence.clone()
    sequence[mask] = tokenizer.mask_token_id

    return sequence

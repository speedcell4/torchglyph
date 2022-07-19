from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Poisson
from transformers import PreTrainedTokenizer
from torchrua import rua_fn


def is_special(input_ids: Tensor, duration: Optional[Tensor], tokenizer: PreTrainedTokenizer) -> Tensor:
    special_tokens = torch.tensor(tokenizer.all_special_ids, dtype=input_ids.dtype, device=input_ids.device)
    mask = (input_ids[..., None] == special_tokens).any(dim=-1)
    if duration is not None:
        mask = torch.segment_reduce(mask.float(), reduce='max', lengths=duration).bool()
    return mask


@rua_fn
def uniform_augment(input_ids: Tensor, tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    mask = torch.rand_like(input_ids, dtype=torch.float32) < p
    mask[is_special(input_ids=input_ids, duration=None, tokenizer=tokenizer)] = False

    input_ids = input_ids.clone()
    input_ids[mask] = tokenizer.mask_token_id

    return input_ids


@rua_fn
def uniform_augment_as_words(input_ids: Tensor, duration: Tensor,
                             tokenizer: PreTrainedTokenizer, p: float) -> Tensor:
    mask = torch.rand_like(duration, dtype=torch.float32) < p
    mask[is_special(input_ids=input_ids, duration=duration, tokenizer=tokenizer)] = False
    mask = torch.repeat_interleave(mask.float(), repeats=duration, dim=0).bool()

    input_ids = input_ids.clone()
    input_ids[mask] = tokenizer.mask_token_id

    return input_ids


@rua_fn
def poisson_augment_as_words(input_ids: Tensor, duration: Tensor,
                             tokenizer: PreTrainedTokenizer, rate: float) -> Tensor:
    rate = torch.scalar_tensor(rate, device=duration.device)
    mask = torch.rand_like(duration, dtype=torch.float32) < Poisson(rate).log_prob(duration).exp()
    mask[is_special(input_ids=input_ids, duration=duration, tokenizer=tokenizer)] = False
    mask = torch.repeat_interleave(mask.float(), repeats=duration, dim=0).bool()

    input_ids = input_ids.clone()
    input_ids[mask] = tokenizer.mask_token_id

    return input_ids

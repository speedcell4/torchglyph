from typing import Tuple

import torch
from torch import Tensor
from torch.types import Device
from transformers import PreTrainedTokenizer


def get_special_ids(tokenizer: PreTrainedTokenizer, device: Device, dtype: torch.dtype = torch.long) -> Tensor:
    return torch.tensor(tokenizer.all_special_ids, device=device, dtype=dtype)


def get_special_mapping(tokenizer: PreTrainedTokenizer, device: Device, dtype: torch.dtype = torch.bool) -> Tensor:
    mapping = torch.zeros((tokenizer.vocab_size,), device=device, dtype=dtype)
    for index in tokenizer.all_special_ids:
        mapping[index] = True
    return mapping


def get_continual_mapping(tokenizer: PreTrainedTokenizer, device: Device, dtype: torch.dtype = torch.bool) -> Tensor:
    mapping = torch.zeros((tokenizer.vocab_size,), device=device, dtype=dtype)
    for token, index in tokenizer.get_vocab().items():
        if token.startswith('##'):
            mapping[index] = True
    return mapping


def mask_mlm_tokens(tokens: Tensor, special_ids: Tensor, tokenizer: PreTrainedTokenizer = None,
                    p_mlm: float = 0.15, p_mask: float = 0.8, p_random: float = 0.1) -> Tuple[Tensor, Tensor, Tensor]:
    ratio = torch.rand_like(tokens)
    ratio[(tokens[..., None] == special_ids).any(dim=-1)] = 1.0

    boundaries = torch.tensor([p_mask, p_mask + p_random, 1.0], device=tokens.device)
    index = torch.bucketize(ratio, boundaries=boundaries * p_mlm)

    mlm_inputs = tokens.clone()
    mlm_inputs[index == 0] = tokenizer.mask_token_id
    mlm_inputs[index == 1] = torch.randint_like(mlm_inputs[index == 1], tokenizer.vocab_size)

    mlm_targets = tokens.clone()
    mlm_targets[index == 3] = tokenizer.pad_token_id

    return mlm_inputs, mlm_targets, index

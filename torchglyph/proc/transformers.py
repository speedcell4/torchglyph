from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import CattedSequence
from torchrua.padding import pad_catted_indices, pad_packed_indices
from transformers import PreTrainedTokenizer, PreTrainedModel

__all__ = [
    'get_special_ids',
    'get_special_mapping',
    'get_continual_mapping',
    'mask_mlm_tokens', 'MaskMlmTokens',
    'mlm_catted_indices', 'mlm_packed_indices',
    'mlm_catted_sequence', 'mlm_packed_sequence', 'mlm_padded_sequence',
]


def get_special_ids(tokenizer: PreTrainedTokenizer,
                    device: Device = None, dtype: torch.dtype = torch.long) -> Tensor:
    return torch.tensor(tokenizer.all_special_ids, device=device, dtype=dtype)


def get_special_mapping(tokenizer: PreTrainedTokenizer,
                        device: Device = None, dtype: torch.dtype = torch.bool) -> Tensor:
    mapping = torch.zeros((tokenizer.vocab_size,), device=device, dtype=dtype)
    for index in tokenizer.all_special_ids:
        mapping[index] = True
    return mapping


def get_continual_mapping(tokenizer: PreTrainedTokenizer,
                          device: Device = None, dtype: torch.dtype = torch.bool) -> Tensor:
    mapping = torch.zeros((tokenizer.vocab_size,), device=device, dtype=dtype)
    for token, index in tokenizer.get_vocab().items():
        if token.startswith('##'):
            mapping[index] = True
    return mapping


def mask_mlm_tokens(tokens: Tensor, tokenizer: PreTrainedTokenizer, special_ids: Tensor = None,
                    p_mlm: float = 0.15, p_mask: float = 0.8, p_random: float = 0.1) -> Tuple[Tensor, Tensor, Tensor]:
    ratio = torch.rand_like(tokens, dtype=torch.float32)
    if special_ids is None:
        special_ids = get_special_ids(tokenizer, device=tokens.device)
    ratio[(tokens[..., None] == special_ids).any(dim=-1)] = 1.0

    boundaries = torch.tensor([p_mask, p_mask + p_random, 1.0], device=tokens.device)
    index = torch.bucketize(ratio, boundaries=boundaries * p_mlm)

    mlm_inputs = tokens.clone()
    mlm_inputs[index == 0] = tokenizer.mask_token_id
    mlm_inputs[index == 1] = torch.randint_like(mlm_inputs[index == 1], tokenizer.vocab_size)

    mlm_targets = tokens.clone()
    mlm_targets[index == 3] = tokenizer.pad_token_id

    return mlm_inputs, mlm_targets, index


class MaskMlmTokens(nn.Module):
    def __init__(self, p_mlm: float = 0.15, p_mask: float = 0.8, p_random: float = 0.1, *,
                 tokenizer: PreTrainedTokenizer) -> None:
        super(MaskMlmTokens, self).__init__()

        self.tokenizer = tokenizer
        self.register_buffer('special_ids', get_special_ids(tokenizer))

        self.p_mlm = p_mlm
        self.p_mask = p_mask
        self.p_random = p_random

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.tokenizer.name_or_path}',
            f'p_mlm={self.p_mlm}',
            f'p_mask={self.p_mask}',
            f'p_random={self.p_random}',
            f'p_keep={1 - self.p_mask - self.p_random}',
        ])

    def forward(self, tokens: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        mlm_inputs, mlm_targets, index = mask_mlm_tokens(
            tokens=tokens, tokenizer=self.tokenizer, special_ids=self.special_ids,
            p_mlm=self.p_mlm, p_mask=self.p_mask, p_random=self.p_random,
        )
        return mlm_inputs, mlm_targets, index


@torch.no_grad()
def mlm_catted_indices(sequence: CattedSequence, tokenizer: PreTrainedTokenizer, device: Device = None):
    if device is None:
        device = sequence.data.device

    size, ptr = pad_catted_indices(
        token_sizes=sequence.token_sizes,
        batch_first=True, device=device,
    )

    input_ids = torch.full(size, fill_value=tokenizer.pad_token_id, device=device, dtype=torch.long)
    input_ids[ptr] = sequence.data
    attention_mask = input_ids != tokenizer.pad_token_id

    return input_ids, attention_mask, ptr


@torch.no_grad()
def mlm_packed_indices(sequence: PackedSequence, tokenizer: PreTrainedTokenizer, device: Device = None):
    if device is None:
        device = sequence.data.device

    size, ptr, _ = pad_packed_indices(
        batch_sizes=sequence.batch_sizes,
        sorted_indices=sequence.sorted_indices,
        unsorted_indices=sequence.unsorted_indices,
        batch_first=True, device=device,
    )

    input_ids = torch.full(size, fill_value=tokenizer.pad_token_id, device=device, dtype=torch.long)
    input_ids[ptr] = sequence.data
    attention_mask = input_ids != tokenizer.pad_token_id

    return input_ids, attention_mask, ptr


def mlm_padded_sequence(sequence: Tensor, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    attention_mask = sequence != tokenizer.pad_token_id
    out_dict = model(input_ids=sequence, attention_mask=attention_mask, return_dict=True)
    return out_dict.last_hidden_state


def mlm_catted_sequence(sequence: CattedSequence, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    input_ids, attention_mask, ptr = mlm_catted_indices(sequence, tokenizer=tokenizer)
    out_dict = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return out_dict.last_hidden_state[ptr]


def mlm_packed_sequence(sequence: PackedSequence, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    input_ids, attention_mask, ptr = mlm_packed_indices(sequence, tokenizer=tokenizer)
    out_dict = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    return out_dict.last_hidden_state[ptr]

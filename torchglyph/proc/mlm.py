from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import CattedSequence, accumulate_sizes
from torchrua.padding import pad_catted_indices, pad_packed_indices
from transformers import PreTrainedTokenizer, PreTrainedModel

__all__ = [
    'get_special_ids',
    'get_special_mapping',
    'get_continual_mapping',
    'mask_mlm_tokens', 'MaskMlmTokens',
    'mlm_catted_indices', 'mlm_packed_indices',
    'mlm_catted_sequence', 'mlm_packed_sequence', 'mlm_padded_sequence',
    'mlm_bag_catted_indices', 'mlm_bag_catted_sequence',
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

    return input_ids, ptr


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

    return input_ids, ptr


def mlm_padded_sequence(sequence: Tensor, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    attention_mask = sequence != tokenizer.pad_token_id
    return model(input_ids=sequence, attention_mask=attention_mask, return_dict=True).last_hidden_state


def mlm_catted_sequence(sequence: CattedSequence, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    sequence, ptr = mlm_catted_indices(sequence=sequence, tokenizer=tokenizer)
    return mlm_padded_sequence(sequence=sequence, tokenizer=tokenizer, model=model)[ptr]


def mlm_packed_sequence(sequence: PackedSequence, tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> Tensor:
    sequence, ptr = mlm_packed_indices(sequence=sequence, tokenizer=tokenizer)
    return mlm_padded_sequence(sequence=sequence, tokenizer=tokenizer, model=model)[ptr]


@torch.no_grad()
def mlm_bag_catted_indices(index: CattedSequence, tokenizer: PreTrainedTokenizer, device: Device = None):
    if device is None:
        device = index.data.device

    (b, t), (batch_ptr, token_ptr) = pad_catted_indices(
        token_sizes=index.token_sizes,
        batch_first=True, device=device,
    )

    input_ids = torch.full((b, t), fill_value=tokenizer.pad_token_id, device=device, dtype=torch.long)
    input_ids[batch_ptr, token_ptr] = index.data

    token_sizes = torch.zeros_like(input_ids)
    token_sizes[batch_ptr, index.data] = 1
    token_sizes = token_sizes.sum(dim=-1)

    indices = index.data + batch_ptr * t
    _, counts = torch.unique(indices, sorted=True, return_counts=True)
    indices = torch.argsort(indices, dim=0, descending=False)
    indices = (token_ptr + batch_ptr * t)[indices]
    offsets = accumulate_sizes(sizes=counts)

    return input_ids, indices, offsets, token_sizes


def mlm_bag_catted_sequence(sequence: CattedSequence, index: CattedSequence, mode: int,
                            tokenizer: PreTrainedTokenizer, model: PreTrainedModel) -> CattedSequence:
    assert torch.equal(sequence.token_sizes, index.token_sizes), f'{sequence.token_sizes} != {index.token_sizes}'

    sequence, indices, offsets, token_sizes = mlm_bag_catted_indices(
        index=index, tokenizer=tokenizer,
        device=sequence.data.device,
    )
    tensor = mlm_padded_sequence(sequence=sequence, tokenizer=tokenizer, model=model)

    data, _, _, _ = torch.embedding_bag(
        tensor.flatten(start_dim=0, end_dim=1),
        indices=indices, offsets=offsets, mode=mode,
    )
    return CattedSequence(data=data, token_sizes=token_sizes)

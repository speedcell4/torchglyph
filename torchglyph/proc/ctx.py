from typing import Any, List

import torch
from torch import Tensor
from torch.types import Device
from torchrua import cat_sequence, token_sizes_to_mask, pad_catted_sequence
from transformers import PreTrainedTokenizer

from torchglyph.proc.abc import Proc

__all__ = [
    'CtxTokenize',
    'CtxCollate',
]


class CtxTokenize(Proc):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 add_special_tokens: bool, prefix: str,
                 dtype: torch.dtype) -> None:
        super(CtxTokenize, self).__init__()

        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.prefix = prefix
        self.dtype = dtype

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.tokenizer.name_or_path}',
            f'add_special_tokens={self.add_special_tokens}',
            f'prefix={self.prefix}',
        ])

    def __call__(self, text: str, **kwargs) -> Any:
        tokens = self.tokenizer.tokenize(text=text, add_special_tokens=self.add_special_tokens)
        ids = self.tokenizer.convert_tokens_to_ids(tokens=tokens)

        return torch.tensor([
            -id if token.startswith(self.prefix) else id
            for token, id in zip(tokens, ids)
        ], dtype=self.dtype)


class CtxCollate(Proc):
    def __init__(self, tokenizer: PreTrainedTokenizer, device: Device) -> None:
        super(CtxCollate, self).__init__()

        self.tokenizer = tokenizer
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tensor], **kwargs):
        sequence, token_sizes = cat_sequence(sequences, device=self.device)

        input_ids = pad_catted_sequence(
            sequence=sequence, token_sizes=token_sizes,
            batch_first=True, padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = token_sizes_to_mask(token_sizes=token_sizes, batch_first=True)

        return dict(
            input_ids=input_ids.abs(),
            attention_mask=attention_mask,
        )

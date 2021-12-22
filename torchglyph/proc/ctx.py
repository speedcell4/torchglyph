from typing import Any

import torch
from transformers import AutoTokenizer, BertTokenizer

from torchglyph.proc.abc import Proc


class CtxTokenize(Proc):
    def __init__(self, pretrained_model_name: str,
                 add_special_tokens: bool, prefix: str,
                 dtype: torch.dtype) -> None:
        super(CtxTokenize, self).__init__()

        self.pretrained_model_name = pretrained_model_name
        self.add_special_tokens = add_special_tokens
        self.prefix = prefix
        self.dtype = dtype

        self.tokenizer: BertTokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def __call__(self, text: str, **kwargs) -> Any:
        tokens = self.tokenizer.tokenize(text=text, add_special_tokens=self.add_special_tokens)
        ids = self.tokenizer.convert_tokens_to_ids(tokens=tokens)

        return torch.tensor([
            -id if token.startswith(self.prefix) else id
            for token, id in zip(tokens, ids)
        ], dtype=self.dtype)

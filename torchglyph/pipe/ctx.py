import torch
from torch.types import Device
from transformers import AutoTokenizer

from torchglyph.pipe import Pipe
from torchglyph.proc.ctx import CtxTokenize, CtxCollate

__all__ = [
    'CtxPipe',
]


class CtxPipe(Pipe):
    def __init__(self, pretrained_model_name: str, device: Device, add_special_tokens: bool = True,
                 prefix: str = '##', dtype: torch.dtype = torch.long) -> None:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        super(CtxPipe, self).__init__(
            pre=CtxTokenize(tokenizer=tokenizer, add_special_tokens=add_special_tokens, prefix=prefix, dtype=dtype),
            vocab=None,
            post=None,
            batch=CtxCollate(tokenizer=tokenizer, device=device),
        )

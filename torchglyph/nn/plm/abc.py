from functools import singledispatch
from logging import getLogger
from typing import List, Union

from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from torchglyph.nn.plm import utils

logger = getLogger(__name__)


@singledispatch
def qof(model: nn.Module, /, **kwargs) -> None:
    raise NotImplementedError(f'{type(model)} is not supported')


class PLM(object):
    mapping = {}
    checkpoints = {}

    def __init__(self, *, lang: str, **kwargs) -> None:
        super(PLM, self).__init__()

        self.lang = lang
        self.pretrained_model_name = self.checkpoints[lang]

        self._config = None
        self._tokenizer = None
        self._model = None

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.pretrained_model_name})'

    @property
    def config(self) -> PretrainedConfig:
        if self._config is None:
            self._config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
            )

        return self._config

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                src_lang=self.mapping.get(self.lang, None),
                use_fast=True,
            )

        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
            )

        return self._model

    def tokenize(self, text: Union[str, List[str]], *,
                 add_prefix_space: bool = False, add_special_tokens: bool = True):
        if isinstance(text, str):
            return utils.tokenize_sequence(
                text=text, tokenizer=self.tokenizer,
                add_prefix_space=add_prefix_space,
                add_special_tokens=add_special_tokens,
            )

        return utils.tokenize_segment(
            text=text, tokenizer=self.tokenizer,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

    def tokenize_batch(self, text: Union[List[str], List[List[str]]], *,
                       add_prefix_space: bool = False, add_special_tokens: bool = True):
        if isinstance(text[0], str):
            return utils.tokenize_sequence_batch(
                text=text, tokenizer=self.tokenizer,
                add_prefix_space=add_prefix_space,
                add_special_tokens=add_special_tokens,
            )

        return utils.tokenize_segment_batch(
            text=text, tokenizer=self.tokenizer,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

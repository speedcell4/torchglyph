import logging
from logging import getLogger
from typing import Union, List

from torch import Tensor
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torchrua import CattedSequence
from transformers import PreTrainedModel, AutoModel
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import PretrainedConfig, AutoConfig

from torchglyph.nn.plm.encode import encode, encode_as_words
from torchglyph.nn.plm.tokenize import tokenize_as_tokens, tokenize_as_words

logger = getLogger(__name__)


class PLM(object):
    def __init__(self, *, lang: str, **kwargs) -> None:
        super(PLM, self).__init__()
        self._lang = lang

    @property
    def pretrained_model_name(self) -> str:
        raise NotImplementedError

    @property
    def lang(self) -> str:
        return self._lang

    @lazy_property
    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
        )

    @lazy_property
    def tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
            add_prefix_space=False,
            src_lang=self.lang,
        )
        logging.info(f'{tokenizer.__class__.__name__}({self.pretrained_model_name}, lang={self.lang})')
        return tokenizer

    @lazy_property
    def model(self) -> PreTrainedModel:
        model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
        )
        logger.info(f'{model.__class__.__name__}({self.pretrained_model_name}, lang={self.lang})')
        return model

    def tokenize_as_tokens(self, sentence: str, sentence_pair: str = None,
                           tokenizer: PreTrainedTokenizer = None, *,
                           max_length: int = None, truncation: bool = True,
                           as_string: bool = False, add_prefix_space: bool = False,
                           add_special_tokens: bool = True):
        return tokenize_as_tokens(
            sentence=sentence, sentence_pair=sentence_pair,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            max_length=max_length, truncation=truncation, as_string=as_string,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

    def tokenize_as_words(self, sentence: List[str],
                          tokenizer: PreTrainedTokenizer = None, *,
                          max_length: int = None, truncation: bool = True,
                          as_string: bool = False, add_prefix_space: bool = False,
                          add_special_tokens: bool = True):
        return tokenize_as_words(
            sentence=sentence,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            max_length=max_length, truncation=truncation, as_string=as_string,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

    def encode(self, input_ids: Union[Tensor, CattedSequence, PackedSequence],
               tokenizer: PreTrainedTokenizer = None, model: PreTrainedModel = None):
        return encode(
            input_ids,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            model=self.model if model is None else model,
        )

    def encode_as_words(self, input_ids: Union[Tensor, CattedSequence, PackedSequence],
                        duration: Union[Tensor, CattedSequence, PackedSequence], reduce: str,
                        tokenizer: PreTrainedTokenizer = None, model: PreTrainedModel = None):
        return encode_as_words(
            input_ids, duration=duration, reduce=reduce,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            model=self.model if model is None else model,
        )

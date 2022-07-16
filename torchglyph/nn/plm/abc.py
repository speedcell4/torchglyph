from typing import Union, List, Optional

from torch import Tensor
from torch.distributions.utils import lazy_property
from torch.nn.utils.rnn import PackedSequence
from torchrua import CattedSequence
from transformers import PreTrainedModel, AutoModel
from transformers import PreTrainedTokenizer, AutoTokenizer
from transformers import PretrainedConfig, AutoConfig

from torchglyph.nn.plm.encode import encode, encode_as_words
from torchglyph.nn.plm.tokenize import tokenize, tokenize_as_words

Sequence = Union[Tensor, CattedSequence, PackedSequence]


class PLM(object):
    def __init__(self, src_lang: str, **kwargs) -> None:
        super(PLM, self).__init__()
        self._src_lang = src_lang

    @property
    def pretrained_model_name(self) -> str:
        raise NotImplementedError

    @property
    def src_lang(self) -> Optional[str]:
        return None

    @property
    def tgt_lang(self) -> Optional[str]:
        return None

    @lazy_property
    def config(self) -> PretrainedConfig:
        return AutoConfig.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name)

    @lazy_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
            src_lang=self.src_lang, tgt_lang=self.tgt_lang, add_prefix_space=False,
        )

    @lazy_property
    def model(self) -> PreTrainedModel:
        return AutoModel.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name)

    def tokenize(self, sentence: str, tokenizer: PreTrainedTokenizer = None, *,
                 as_string: bool = False, add_prefix_space: bool = False, add_special_tokens: bool = True):
        input_ids = tokenize(
            sentence=sentence,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

        return self.tokenizer.convert_ids_to_tokens(input_ids) if as_string else input_ids

    def tokenize_as_words(self, sentence: List[str], tokenizer: PreTrainedTokenizer = None, *,
                          as_string: bool = False, add_prefix_space: bool = False, add_special_tokens: bool = True):
        input_ids, duration = tokenize_as_words(
            sentence=sentence,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

        return self.tokenizer.convert_ids_to_tokens(input_ids) if as_string else input_ids, duration

    def encode(self, input_ids: Sequence,
               tokenizer: PreTrainedTokenizer = None, model: PreTrainedModel = None):
        return encode(
            input_ids=input_ids,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            model=self.model if model is None else model,
        )

    def encode_as_words(self, input_ids: Sequence, duration: Sequence,
                        tokenizer: PreTrainedTokenizer = None, model: PreTrainedModel = None):
        return encode_as_words(
            input_ids=input_ids, duration=duration,
            tokenizer=self.tokenizer if tokenizer is None else tokenizer,
            model=self.model if model is None else model,
        )


class RoBertaBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'roberta-base',
            'de': 'uklfr/gottbert-base',
            'fr': 'camembert-base',
            'zh': 'hfl/chinese-macbert-base',
            'ja': 'rinna/japanese-roberta-base',
        }[self._src_lang]

    @property
    def src_lang(self) -> str:
        return {
            'en': 'en_XX',
            'de': 'de_DE',
            'fr': 'fr_XX',
            'zh': 'zh_CN',
            'ja': 'ja_XX',
        }[self._src_lang]


class BartBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'facebook/bart-base',
            'fr': 'moussaKam/barthez',
            'zh': 'fnlp/bart-base-chinese',
        }[self._src_lang]

    @property
    def src_lang(self) -> str:
        return {
            'en': 'en_XX',
            'fr': 'fr_XX',
            'zh': 'zh_CN',
        }[self._src_lang]

from typing import Union, List

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
    def __init__(self, lang: str, **kwargs) -> None:
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
        return AutoConfig.from_pretrained(pretrained_model_name_or_path=self.pretrained_model_name)

    @lazy_property
    def tokenizer(self) -> PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.pretrained_model_name,
            add_prefix_space=False,
            src_lang=self.lang,
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
        }[self._lang]


class BartBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'facebook/bart-base',
            'fr': 'moussaKam/barthez',
            'zh': 'fnlp/bart-base-chinese',
        }[self._lang]


class MBartLarge(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'ar': 'facebook/mbart-large-cc25',
            'cs': 'facebook/mbart-large-cc25',
            'de': 'facebook/mbart-large-cc25',
            'en': 'facebook/mbart-large-cc25',
            'es': 'facebook/mbart-large-cc25',
            'et': 'facebook/mbart-large-cc25',
            'fi': 'facebook/mbart-large-cc25',
            'fr': 'facebook/mbart-large-cc25',
            'gu': 'facebook/mbart-large-cc25',
            'hi': 'facebook/mbart-large-cc25',
            'it': 'facebook/mbart-large-cc25',
            'ja': 'facebook/mbart-large-cc25',
            'kk': 'facebook/mbart-large-cc25',
            'ko': 'facebook/mbart-large-cc25',
            'lt': 'facebook/mbart-large-cc25',
            'lv': 'facebook/mbart-large-cc25',
            'my': 'facebook/mbart-large-cc25',
            'ne': 'facebook/mbart-large-cc25',
            'nl': 'facebook/mbart-large-cc25',
            'ro': 'facebook/mbart-large-cc25',
            'ru': 'facebook/mbart-large-cc25',
            'si': 'facebook/mbart-large-cc25',
            'tr': 'facebook/mbart-large-cc25',
            'vi': 'facebook/mbart-large-cc25',
            'zh': 'facebook/mbart-large-cc25',
        }[self._lang]

    @property
    def lang(self) -> str:
        return {
            'ar': 'ar_AR',
            'cs': 'cs_CZ',
            'de': 'de_DE',
            'en': 'en_XX',
            'es': 'es_XX',
            'et': 'et_EE',
            'fi': 'fi_FI',
            'fr': 'fr_XX',
            'gu': 'gu_IN',
            'hi': 'hi_IN',
            'it': 'it_IT',
            'ja': 'ja_XX',
            'kk': 'kk_KZ',
            'ko': 'ko_KR',
            'lt': 'lt_LT',
            'lv': 'lv_LV',
            'my': 'my_MM',
            'ne': 'ne_NP',
            'nl': 'nl_XX',
            'ro': 'ro_RO',
            'ru': 'ru_RU',
            'si': 'si_LK',
            'tr': 'tr_TR',
            'vi': 'vi_VN',
            'zh': 'zh_CN',
        }[self._lang]

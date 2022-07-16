from typing import List

from transformers import PreTrainedTokenizer

from torchglyph.nn.plm.abc import PLM


class BertBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'bert-base-cased',
            'de': 'bert-base-german-cased',
        }[self._lang]


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

    def tokenize(self, sentence: str, tokenizer: PreTrainedTokenizer = None, *,
                 as_string: bool = False, add_prefix_space: bool = False, add_special_tokens: bool = True):
        if self.lang == 'ja' and add_special_tokens:
            sentence = f'[CLS]{sentence}'

        return super(RoBertaBase, self).tokenize(
            sentence, tokenizer, as_string=as_string,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )

    def tokenize_as_words(self, sentence: List[str], tokenizer: PreTrainedTokenizer = None, *,
                          as_string: bool = False, add_prefix_space: bool = False, add_special_tokens: bool = True):
        if self.lang == 'ja' and add_special_tokens:
            sentence = ['[CLS]', *sentence]

        return super(RoBertaBase, self).tokenize_as_words(
            sentence, tokenizer, as_string=as_string,
            add_prefix_space=add_prefix_space,
            add_special_tokens=add_special_tokens,
        )


class BartBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'facebook/bart-base',
            'fr': 'moussaKam/barthez',
            'zh': 'fnlp/bart-base-chinese',
        }[self._lang]

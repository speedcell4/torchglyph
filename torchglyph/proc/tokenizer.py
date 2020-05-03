from typing import Union, List

import transformers
from allennlp.data import Token as AllenToken, Instance as AllenInstance
from allennlp.data.fields import TextField as AllenTextField
from allennlp.data.token_indexers import ELMoTokenCharactersIndexer

from torchglyph.io import toggle_loggers
from torchglyph.proc import Proc

toggle_loggers('allennlp', False)
toggle_loggers('transformers', False)


class ELMoTokenizer(Proc):
    def __init__(self) -> None:
        super(ELMoTokenizer, self).__init__()
        self.tokenizer = ELMoTokenCharactersIndexer()

    def __call__(self, data: List[str], *args, **kwargs):
        data = [AllenToken(token) for token in data]
        return AllenInstance({"elmo": AllenTextField(data, {'character_ids': self.tokenizer})})


class TransformerTokenizerProc(Proc):
    def __init__(self, weight: str) -> None:
        super(TransformerTokenizerProc, self).__init__()
        self.weightt = weight

    def extra_repr(self) -> str:
        return f'weight={self.weight}'

    def __call__(self, data: Union[str, List[str]], **kwargs) -> List[int]:
        if not isinstance(data, str):
            data = ' '.join(data)
        return self.tokenizer.encode(data)


class BertTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'bert-base-uncased') -> None:
        super(BertTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(weight)


class OpenAIGPTTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'openai-gpt') -> None:
        super(OpenAIGPTTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.OpenAIGPTTokenizer.from_pretrained(weight)


class GPT2Tokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'gpt2') -> None:
        super(GPT2Tokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained(weight)


class CTRLTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'ctrl') -> None:
        super(CTRLTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.CTRLTokenizer.from_pretrained(weight)


class TransfoXLTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'transfo-xl-wt103') -> None:
        super(TransfoXLTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.TransfoXLTokenizer.from_pretrained(weight)


class XLNetTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'xlnet-base-cased') -> None:
        super(XLNetTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.XLNetTokenizer.from_pretrained(weight)


class XLMTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'xlm-mlm-enfr-1024') -> None:
        super(XLMTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.XLMTokenizer.from_pretrained(weight)


class DistilBertTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'distilbert-base-cased') -> None:
        super(DistilBertTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained(weight)


class RobertaTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'roberta-base') -> None:
        super(RobertaTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.RobertaTokenizer.from_pretrained(weight)


class XLMRobertaTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'xlm-roberta-base') -> None:
        super(XLMRobertaTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(weight)


class BartTokenizer(TransformerTokenizerProc):
    def __init__(self, weight: str = 'bart-large') -> None:
        super(BartTokenizer, self).__init__(weight=weight)
        self.tokenizer = transformers.BartTokenizer.from_pretrained(weight)

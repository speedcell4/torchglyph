from typing import Union, List

import transformers

from torchglyph.pipe import Proc


class TokenizerProc(Proc):
    def __init__(self, weights: str) -> None:
        super(TokenizerProc, self).__init__()
        self.weights = weights

    def extra_repr(self) -> str:
        return f'weights={self.weights}'

    def __call__(self, data: Union[str, List[str]], **kwargs) -> List[int]:
        if not isinstance(data, str):
            data = ' '.join(data)
        return self.tokenizer.encode(data)


class BertTokenizer(TokenizerProc):
    def __init__(self, weights: str = 'bert-base-uncased'):
        super(BertTokenizer, self).__init__(weights=weights)
        self.tokenizer = transformers.BertTokenizer.from_pretrained(weights)


class CTRLTokenizer(TokenizerProc):
    def __init__(self, weights: str = 'ctrl'):
        super(CTRLTokenizer, self).__init__(weights=weights)
        self.tokenizer = transformers.CTRLTokenizer.from_pretrained(weights)


class XLNetTokenizer(TokenizerProc):
    def __init__(self, weights: str = 'xlnet-base-cased'):
        super(XLNetTokenizer, self).__init__(weights=weights)
        self.tokenizer = transformers.XLNetTokenizer.from_pretrained(weights)


class XLMTokenizer(TokenizerProc):
    def __init__(self, weights: str = 'xlm-mlm-enfr-1024'):
        super(XLMTokenizer, self).__init__(weights=weights)
        self.tokenizer = transformers.XLMTokenizer.from_pretrained(weights)


class BartTokenizer(TokenizerProc):
    def __init__(self, weights: str = 'bart-large'):
        super(BartTokenizer, self).__init__(weights=weights)
        self.tokenizer = transformers.BartTokenizer.from_pretrained(weights)

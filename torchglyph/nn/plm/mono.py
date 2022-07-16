from torchglyph.nn.plm.abc import PLM


class BertBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'bert-base-cased',
            'de': 'bert-base-german-cased',
        }[self._lang]


class RoBERTaBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'roberta-base',
            'de': 'uklfr/gottbert-base',
            'fr': 'camembert-base',
            'zh': 'hfl/chinese-macbert-base',
        }[self._lang]


class XLNetBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'xlnet-base-cased',
            'zh': 'hfl/chinese-xlnet-base',
        }[self._lang]


class XLNetLarge(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'xlnet-large-cased',
            'zh': 'hfl/chinese-xlnet-large',
        }[self._lang]


class BartBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'facebook/bart-base',
            'fr': 'moussaKam/barthez',
            'zh': 'fnlp/bart-base-chinese',
        }[self._lang]


class DeBERTaBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return {
            'en': 'microsoft/deberta-base',
        }[self._lang]

from torchglyph.nn.plm.abc import PLM


class MultiBertBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return 'bert-base-multilingual-cased'


class MultiRobertaBase(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return 'xlm-roberta-base'


class MultiBartLarge(PLM):
    @property
    def pretrained_model_name(self) -> str:
        return 'facebook/mbart-large-cc25'

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

from collections import defaultdict

from torchglyph.nn.plm import PLM


class BertBase(PLM):
    checkpoints = {
        'en': 'bert-base-cased',
        'de': 'bert-base-german-cased',
    }


class BertLarge(PLM):
    checkpoints = {
        'en': 'bert-large-cased',
    }


class MBertBase(PLM):
    checkpoints = defaultdict(lambda: 'bert-base-multilingual-cased')


class RobertaBase(PLM):
    checkpoints = {
        'en': 'roberta-base',
        'de': 'uklfr/gottbert-base',
        'fr': 'camembert-base',
        'zh': 'hfl/chinese-macbert-base',
    }


class RobertaLarge(PLM):
    checkpoints = {
        'en': 'roberta-large',
        'fr': 'camembert-large',
        'zh': 'hfl/chinese-macbert-large',
    }


class Gpt2(PLM):
    checkpoints = {
        'en': 'gpt2',
        'de': 'dbmdz/german-gpt2',
    }


class Gpt2Medium(PLM):
    checkpoints = {
        'en': 'gpt2-medium'
    }


class Gpt2Large(PLM):
    checkpoints = {
        'en': 'gpt2-large'
    }


class Gpt2XL(PLM):
    checkpoints = {
        'en': 'gpt2-xl'
    }


class XlmRobertaBase(PLM):
    checkpoints = defaultdict(lambda: 'xlm-roberta-base')


class XLNetBase(PLM):
    checkpoints = {
        'en': 'xlnet-base-cased',
        'zh': 'hfl/chinese-xlnet-base',
    }


class XLNetLarge(PLM):
    checkpoints = {
        'en': 'xlnet-large-cased',
        'zh': 'hfl/chinese-xlnet-large',
    }


class BartBase(PLM):
    checkpoints = {
        'en': 'facebook/bart-base',
        'fr': 'moussaKam/barthez',
        'zh': 'fnlp/bart-base-chinese',
    }


class BartLarge(PLM):
    checkpoints = {
        'en': 'facebook/bart-large',
    }


class MBartLarge(PLM):
    mapping = {
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
    }

    checkpoints = defaultdict(lambda: 'facebook/mbart-large-cc25')


class DeBERTaBase(PLM):
    checkpoints = {
        'en': 'microsoft/deberta-base',
    }

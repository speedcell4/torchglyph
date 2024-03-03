from collections import defaultdict

from transformers.models.bart.modeling_bart import BartDecoderLayer, BartEncoderLayer, BartModel

from torchglyph.nn.plm.abc import PLM, qof


@qof.register
def qof_bart(model: BartModel, /, ) -> None:
    model.requires_grad_(False)

    for layer in model.encoder.layers:  # type: BartEncoderLayer
        layer.self_attn.q_proj.requires_grad_(True)
        layer.self_attn.out_proj.requires_grad_(True)

        layer.fc1.requires_grad_(True)
        layer.fc2.requires_grad_(True)

    for layer in model.decoder.layers:  # type: BartDecoderLayer
        layer.self_attn.q_proj.requires_grad_(True)
        layer.self_attn.out_proj.requires_grad_(True)

        layer.encoder_attn.q_proj.requires_grad_(True)
        layer.encoder_attn.out_proj.requires_grad_(True)

        layer.fc1.requires_grad_(True)
        layer.fc2.requires_grad_(True)


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

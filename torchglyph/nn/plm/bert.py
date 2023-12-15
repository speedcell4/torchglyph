from collections import defaultdict

from transformers.models.bert.modeling_bert import BertLayer, BertModel

from torchglyph.nn.plm.abc import PLM, qof


@qof.register
def qof_bert(model: BertModel):
    model.requires_grad_(False)

    for layer in model.encoder.layer:  # type: BertLayer
        layer.attention.self.query.requires_grad_(True)
        layer.attention.output.requires_grad_(True)

        if layer.add_cross_attention:
            layer.crossattention.self.query.requires_grad_(True)
            layer.crossattention.output.requires_grad_(True)

        layer.intermediate.dense.requires_grad_(True)
        layer.output.dense.requires_grad_(True)


class BertBase(PLM):
    checkpoints = {
        'en': 'bert-base-cased',
        'de': 'bert-base-german-cased',
        'zh': 'bert-base-chinese',
        'ja': 'cl-tohoku/bert-base-japanese',
    }


class BertLarge(PLM):
    checkpoints = {
        'en': 'bert-large-cased',
    }


class MBertBase(PLM):
    checkpoints = defaultdict(lambda: 'bert-base-multilingual-cased')

from collections import defaultdict

from transformers.models.roberta.modeling_roberta import RobertaLayer, RobertaModel

from torchglyph.nn.plm.abc import PLM, qof


@qof.register
def qof_roberta(model: RobertaModel, /, ) -> None:
    model.requires_grad_(False)

    for layer in model.encoder.layer:  # type: RobertaLayer
        layer.attention.self.query.requires_grad_(True)
        layer.attention.output.dense.requires_grad_(True)

        if layer.add_cross_attention:
            layer.crossattention.self.query.requires_grad_(True)
            layer.crossattention.output.requires_grad_(True)

        layer.intermediate.dense.requires_grad_(True)
        layer.output.dense.requires_grad_(True)


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


class XlmRobertaBase(PLM):
    checkpoints = defaultdict(lambda: 'xlm-roberta-base')

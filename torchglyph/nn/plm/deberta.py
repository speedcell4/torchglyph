from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Layer, DebertaV2Model

from torchglyph.nn.plm.abc import qof


@qof.register
def qof_deberta(model: DebertaV2Model, /, ) -> None:
    model.requires_grad_(False)

    for layer in model.encoder.layer:  # type: DebertaV2Layer
        layer.attention.self.query_proj.requires_grad_(True)
        layer.attention.output.dense.requires_grad_(True)

        layer.intermediate.dense.requires_grad_(True)
        layer.output.dense.requires_grad_(True)

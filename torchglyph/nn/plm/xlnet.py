from transformers.models.xlnet.modeling_xlnet import XLNetLayer, XLNetModel

from torchglyph.nn.plm.abc import PLM, qof


@qof.register
def qof_xlnet(model: XLNetModel) -> None:
    model.requires_grad_(False)

    for layer in model.layer:  # type: XLNetLayer
        layer.rel_attn.q.requires_grad_(True)
        layer.rel_attn.o.requires_grad_(True)

        layer.rel_attn.r_r_bias.requires_grad_(True)
        layer.rel_attn.r_s_bias.requires_grad_(True)
        layer.rel_attn.r_s_bias.requires_grad_(True)
        layer.rel_attn.seg_embed.requires_grad_(True)

        layer.ff.layer_1.requires_grad_(True)
        layer.ff.layer_2.requires_grad_(True)


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

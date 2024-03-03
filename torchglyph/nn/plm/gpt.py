from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Model

from torchglyph.nn.plm.abc import PLM, qof


@qof.register
def qof_gpt(model: GPT2Model, /, ) -> None:
    model.requires_grad_(False)

    for layer in model.h:  # type: GPT2Block
        if layer.attn.is_cross_attention:
            layer.attn.q_attn.requires_grad_(True)
            layer.attn.c_proj.requires_grad_(True)
        else:
            layer.attn.c_attn.requires_grad_(True)

        if hasattr(layer, 'crossattention'):
            if layer.crossattention.is_cross_attention:
                layer.crossattention.q_attn.requires_grad_(True)
                layer.crossattention.c_proj.requires_grad_(True)
            else:
                layer.crossattention.c_attn.requires_grad_(True)

        layer.mlp.c_fc.requires_grad_(True)
        layer.mlp.c_proj.requires_grad_(True)


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

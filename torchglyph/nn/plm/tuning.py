from functools import singledispatch

from torch import nn
from transformers.models.bart.modeling_bart import BartAttention, BartDecoderLayer, BartEncoderLayer, BartModel
from transformers.models.bert.modeling_bert import BertIntermediate, BertModel, BertOutput, BertSelfAttention, \
    BertSelfOutput
from transformers.models.mbart.modeling_mbart import MBartAttention, MBartDecoderLayer, MBartEncoderLayer, MBartModel
from transformers.models.roberta.modeling_roberta import RobertaIntermediate, RobertaModel, RobertaOutput, \
    RobertaSelfAttention, RobertaSelfOutput


def full(*, self: nn.Module, **kwargs):
    self.requires_grad_(True)


def qof(*, self: nn.Module, **kwargs):
    self.requires_grad_(False)
    qof_recur(self, **kwargs)


@singledispatch
def qof_recur(self: nn.Module, **kwargs):
    pass


@qof_recur.register
def qof_bert_model(self: BertModel, **kwargs):
    for module in self.modules():
        if self is not module:
            qof_recur(module, **kwargs)


@qof_recur.register
def qof_bert_self_attention(self: BertSelfAttention, **kwargs):
    self.query.requires_grad_(True)


@qof_recur.register
def qof_bert_self_output(self: BertSelfOutput, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_bert_intermediate(self: BertIntermediate, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_bert_output(self: BertOutput, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_roberta_model(self: RobertaModel, **kwargs):
    for module in self.modules():
        if self is not module:
            qof_recur(module, **kwargs)


@qof_recur.register
def qof_roberta_self_attention(self: RobertaSelfAttention, **kwargs):
    self.query.requires_grad_(True)


@qof_recur.register
def qof_roberta_self_output(self: RobertaSelfOutput, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_roberta_intermediate(self: RobertaIntermediate, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_roberta_output(self: RobertaOutput, **kwargs):
    self.dense.requires_grad_(True)


@qof_recur.register
def qof_bart_model(self: BartModel, **kwargs):
    for module in self.modules():
        if self is not module:
            qof_recur(module, **kwargs)


@qof_recur.register
def qof_bart_attention(self: BartAttention, **kwargs):
    self.q_proj.requires_grad_(True)
    self.out_proj.requires_grad_(True)


@qof_recur.register
def qof_bart_encoder_layer(self: BartEncoderLayer, **kwargs):
    self.fc1.requires_grad_(True)
    self.fc2.requires_grad_(True)


@qof_recur.register
def qof_bart_decoder_layer(self: BartDecoderLayer, **kwargs):
    self.fc1.requires_grad_(True)
    self.fc2.requires_grad_(True)


@qof_recur.register
def qof_mbart_model(self: MBartModel, **kwargs):
    for module in self.modules():
        if self is not module:
            qof_recur(module, **kwargs)


@qof_recur.register
def qof_mbart_attention(self: MBartAttention, **kwargs):
    self.q_proj.requires_grad_(True)
    self.out_proj.requires_grad_(True)


@qof_recur.register
def qof_mbart_encoder_layer(self: MBartEncoderLayer, **kwargs):
    self.fc1.requires_grad_(True)
    self.fc2.requires_grad_(True)


@qof_recur.register
def qof_mbart_decoder_layer(self: MBartDecoderLayer, **kwargs):
    self.fc1.requires_grad_(True)
    self.fc2.requires_grad_(True)

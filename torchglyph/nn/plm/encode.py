from functools import singledispatch
from typing import Union

from torch import Tensor
from torchrua import cat_padded_sequence, pad_catted_sequence, CattedSequence
from torchrua import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torchrua import segment_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer

Sequence = Union[Tensor, CattedSequence, PackedSequence]


@singledispatch
def encode_as_tokens(input_ids: Sequence, token_type_ids: Sequence = None, *,
                     model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Sequence:
    out = model(
        input_ids=input_ids,
        attention_mask=input_ids != tokenizer.pad_token_id,
        token_type_ids=token_type_ids,
        return_dict=True,
    )
    return out.last_hidden_state


@encode_as_tokens.register
def encode_catted_sequence_as_tokens(input_ids: CattedSequence,
                                     token_type_ids: CattedSequence = None, *,
                                     model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    input_ids, token_sizes = pad_catted_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if token_type_ids is not None:
        token_type_ids, _ = pad_catted_sequence(token_type_ids, batch_first=True, padding_value=0)

    return cat_padded_sequence(
        sequence=encode_as_tokens(input_ids, token_type_ids=token_type_ids, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@encode_as_tokens.register
def encode_packed_sequence_tokens(input_ids: PackedSequence,
                                  token_type_ids: PackedSequence = None, *,
                                  model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    input_ids, token_sizes = pad_packed_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    if token_type_ids is not None:
        token_type_ids, _ = pad_packed_sequence(token_type_ids, batch_first=True, padding_value=0)

    return pack_padded_sequence(
        sequence=encode_as_tokens(input_ids, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@singledispatch
def encode_as_words(input_ids: Sequence, duration: Sequence, reduce: str,
                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Sequence:
    raise NotImplementedError


@encode_as_words.register
def encode_catted_sequence_as_words(input_ids: CattedSequence, duration: CattedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    input_ids, token_sizes = pad_catted_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode_as_tokens(input_ids, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )


@encode_as_words.register
def encode_packed_sequence_as_words(input_ids: PackedSequence, duration: PackedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    input_ids, token_sizes = pad_packed_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode_as_tokens(input_ids, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )

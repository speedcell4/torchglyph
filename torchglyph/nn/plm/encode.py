from functools import singledispatch

from torch import Tensor
from torchrua import cat_padded_sequence, pad_catted_sequence, CattedSequence
from torchrua import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torchrua import segment_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer


@singledispatch
def encode(input_ids: Tensor, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tensor:
    out = model(
        input_ids=input_ids,
        attention_mask=input_ids != tokenizer.pad_token_id,
        return_dict=True,
    )
    return out.last_hidden_state


@encode.register
def encode_catted_sequence(input_ids: CattedSequence,
                           model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    input_ids, token_sizes = pad_catted_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return cat_padded_sequence(
        sequence=encode(input_ids, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@encode.register
def encode_packed_sequence(input_ids: PackedSequence,
                           model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    input_ids, token_sizes = pad_packed_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return pack_padded_sequence(
        sequence=encode(input_ids, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@singledispatch
def encode_as_words(input_ids: Tensor, duration: Tensor, reduce: str,
                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tensor:
    raise NotImplementedError


@encode_as_words.register
def encode_catted_sequence_as_words(input_ids: CattedSequence, duration: CattedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    input_ids, token_sizes = pad_catted_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode(input_ids, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )


@encode_as_words.register
def encode_packed_sequence_as_words(input_ids: PackedSequence, duration: PackedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    input_ids, token_sizes = pad_packed_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode(input_ids, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )

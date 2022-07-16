from functools import singledispatch

from torch import Tensor
from torchrua import cat_padded_sequence, pad_catted_sequence, CattedSequence
from torchrua import pad_packed_sequence, pack_padded_sequence, PackedSequence
from torchrua import segment_sequence
from transformers import PreTrainedModel, PreTrainedTokenizer


@singledispatch
def encode(sequence: Tensor, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tensor:
    out = model(
        input_ids=sequence,
        attention_mask=sequence != tokenizer.pad_token_id,
        return_dict=True,
    )
    return out.last_hidden_state


@encode.register
def encode_catted_sequence(sequence: CattedSequence,
                           model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    sequence, token_sizes = pad_catted_sequence(sequence, batch_first=True, padding_value=tokenizer.pad_token_id)
    return cat_padded_sequence(
        sequence=encode(sequence, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@encode.register
def encode_packed_sequence(sequence: PackedSequence,
                           model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    sequence, token_sizes = pad_packed_sequence(sequence, batch_first=True, padding_value=tokenizer.pad_token_id)
    return pack_padded_sequence(
        sequence=encode(sequence, model=model, tokenizer=tokenizer),
        token_sizes=token_sizes, batch_first=True,
    )


@singledispatch
def encode_as_words(sequence: Tensor, duration: Tensor,
                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tensor:
    raise NotImplementedError


@encode_as_words.register
def encode_catted_sequence_as_words(sequence: CattedSequence, duration: CattedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> CattedSequence:
    sequence, token_sizes = pad_catted_sequence(sequence, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode(sequence, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )


@encode_as_words.register
def encode_packed_sequence_as_words(sequence: PackedSequence, duration: PackedSequence, reduce: str,
                                    model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> PackedSequence:
    sequence, token_sizes = pad_packed_sequence(sequence, batch_first=True, padding_value=tokenizer.pad_token_id)
    return segment_sequence(
        tensor=encode(sequence, model=model, tokenizer=tokenizer),
        sizes=duration, reduce=reduce, batch_first=True,
    )

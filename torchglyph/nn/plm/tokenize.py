from typing import List, Tuple

from transformers import PreTrainedTokenizer


def tokenize(sequence: str, tokenizer: PreTrainedTokenizer,
             add_prefix_space: bool = False, add_special_tokens: bool = True) -> List[int]:
    return tokenizer(
        f' {sequence}' if add_prefix_space else sequence,
        add_special_tokens=add_special_tokens,
        is_split_into_words=False,
        return_special_tokens_mask=False,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_length=False,
        return_tensors=None,
    )['input_ids']


def tokenize_as_words(sequence: List[str], tokenizer: PreTrainedTokenizer,
                      add_prefix_space: bool = False, add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
    input_ids = tokenizer.batch_encode_plus(
        [f' {word}' if index > 0 or add_prefix_space else word for index, word in enumerate(sequence)],
        add_special_tokens=False,
        is_split_into_words=False,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_length=False,
        return_tensors=None,
    )['input_ids']

    if add_special_tokens:
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    sequence, duration = [], []
    for input_id in input_ids:
        if isinstance(input_id, int):
            sequence.append(input_id)
            duration.append(1)
        elif isinstance(input_id, (set, list, tuple)):
            sequence.extend(input_id)
            duration.append(len(input_id))
        else:
            raise TypeError(f'the type {type(input_id)} of {input_id} is not supported')

    return sequence, duration

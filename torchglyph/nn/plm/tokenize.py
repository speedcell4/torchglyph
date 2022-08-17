from typing import List, Tuple, Optional

from transformers import PreTrainedTokenizer


def tokenize_as_tokens(sentence: str, sentence_pair: Optional[str],
                       tokenizer: PreTrainedTokenizer,
                       max_length: int = None, truncation: bool = True,
                       as_string: bool = False,
                       add_prefix_space: bool = False,
                       add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
    out = tokenizer(
        f' {sentence}' if add_prefix_space else sentence, sentence_pair,
        add_special_tokens=add_special_tokens,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        truncation=truncation,
        is_split_into_words=False,
        return_special_tokens_mask=False,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=True,
        return_attention_mask=False,
        return_length=False,
        return_tensors=None,
    )

    token_ids = out['input_ids']
    token_type_ids = out['token_type_ids']

    if as_string:
        token_ids = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    assert len(token_ids) == len(token_type_ids)

    return token_ids, token_type_ids


def tokenize_as_words(sentence: List[str],
                      tokenizer: PreTrainedTokenizer,
                      max_length: int = None, truncation: bool = True,
                      as_string: bool = False,
                      add_prefix_space: bool = False,
                      add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
    out = tokenizer(
        [f' {word}' if index > 0 or add_prefix_space else word for index, word in enumerate(sentence)],
        add_special_tokens=False,
        max_length=tokenizer.model_max_length if max_length is None else max_length,
        truncation=truncation,
        is_split_into_words=False,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_length=False,
        return_tensors=None,
    )

    input_ids = out['input_ids']

    if add_special_tokens:
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    token_ids, duration = [], []
    for input_id in input_ids:
        if isinstance(input_id, int):
            token_ids.append(input_id)
            duration.append(1)
        elif isinstance(input_id, (set, list, tuple)):
            token_ids.extend(input_id)
            duration.append(len(input_id))
        else:
            raise TypeError(f'the type {type(input_id)} of {input_id} is not supported')

    if as_string:
        token_ids = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    return token_ids, duration

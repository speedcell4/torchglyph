from typing import List, Tuple

from transformers import PreTrainedTokenizer


def tokenize(sentence: str, tokenizer: PreTrainedTokenizer,
             add_prefix_space: bool = False, add_special_tokens: bool = True) -> List[int]:
    sentence = tokenizer(
        f' {sentence}' if add_prefix_space else sentence,
        add_special_tokens=add_special_tokens,
        is_split_into_words=False,
        return_special_tokens_mask=False,
        return_overflowing_tokens=False,
        return_offsets_mapping=False,
        return_token_type_ids=False,
        return_attention_mask=False,
        return_length=False,
        return_tensors=None,
    )
    return sentence['input_ids']


def tokenize_as_words(sentence: List[str], tokenizer: PreTrainedTokenizer,
                      add_prefix_space: bool = False, add_special_tokens: bool = True) -> Tuple[List[int], List[int]]:
    sentence = tokenizer(
        [f' {word}' if index > 0 or add_prefix_space else word for index, word in enumerate(sentence)],
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
        sentence = tokenizer.build_inputs_with_special_tokens(sentence)

    input_ids, duration = [], []
    for item in sentence:
        if isinstance(item, int):
            input_ids.append(item)
            duration.append(1)
        elif isinstance(item, (set, list, tuple)):
            input_ids.extend(item)
            duration.append(len(item))
        else:
            raise TypeError(f'the type {type(item)} of {item} is not supported')

    return input_ids, duration

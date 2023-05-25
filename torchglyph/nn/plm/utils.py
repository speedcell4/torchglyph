from typing import List, Union

from transformers import PreTrainedTokenizer


def tokenize_sequence(text: str, *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                      add_prefix_space: bool = False, add_special_tokens: bool = True):
    return tokenizer(
        f' {text}' if add_prefix_space else text,
        add_special_tokens=add_special_tokens,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']


def tokenize_sequence_batch(text: List[str], *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                            add_prefix_space: bool = False, add_special_tokens: bool = True) -> List[List[int]]:
    return tokenizer(
        [f' {sequence}' for sequence in text] if add_prefix_space else text,
        add_special_tokens=add_special_tokens,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']


def add_prefix(sequence: List[str], add_prefix_space: bool):
    return [f' {token}' if index > 0 or add_prefix_space else token for index, token in enumerate(sequence)]


def postprocess(input_ids: List[Union[int, List[int]]]):
    out1, out2 = [], []

    for input_id in input_ids:
        if isinstance(input_id, int):
            out1.append(input_id)
            out2.append(1)
        else:
            out1.extend(input_id)
            out2.append(len(input_id))

    return out1, out2


def tokenize_segment(text: List[str], *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                     add_prefix_space: bool = False, add_special_tokens: bool = True):
    input_ids = tokenizer(
        add_prefix(text, add_prefix_space=add_prefix_space),
        add_special_tokens=False,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']

    if add_special_tokens:
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

    return postprocess(input_ids)


def tokenize_segment_batch(text: List[List[str]], *, tokenizer: PreTrainedTokenizer, max_length: int = None,
                           add_prefix_space: bool = False, add_special_tokens: bool = True):
    input_ids_batch = tokenizer(
        [token for sequence in text for token in add_prefix(sequence, add_prefix_space=add_prefix_space)],
        add_special_tokens=False,
        truncation=True, max_length=max_length or tokenizer.model_max_length,
        return_tensors=None,
        return_attention_mask=False,
    )['input_ids']

    start, end, out1, out2 = 0, 0, [], []
    for sequence in text:
        start, end = end, end + len(sequence)

        input_ids = input_ids_batch[start:end]
        if add_special_tokens:
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        o1, o2 = postprocess(input_ids)
        out1.append(o1)
        out2.append(o2)

    return out1, out2

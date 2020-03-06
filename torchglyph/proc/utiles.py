from typing import Union

from torchglyph.vocab import Vocab


def stoi(token: Union[str, int], vocab: Vocab) -> int:
    if isinstance(token, str):
        assert vocab is not None, f'{Vocab.__name__} is not built yet'
        assert token in vocab.stoi, f'{token} is not in {Vocab.__name__}'

        return vocab.stoi[token]
    else:
        return token

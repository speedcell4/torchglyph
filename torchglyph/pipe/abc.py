from abc import ABCMeta
from collections import Counter
from typing import Optional, Union, List, Any, Tuple

from torchglyph.proc import Proc, ProcList, compress, subs, Identity
from torchglyph.vocab import Vocab

__all__ = [
    'THRESHOLD',
    'Pipe', 'RawPipe',
]

THRESHOLD = 8


class Pipe(object, metaclass=ABCMeta):
    def __init__(self, pre: ProcList = None, vocab: ProcList = None, post: ProcList = None,
                 batch: ProcList = None) -> None:
        super(Pipe, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self._pre_proc = Proc.from_list(compress(processors=pre))
        self._vocab_proc = Proc.from_list(compress(processors=vocab))
        self._post_proc = Proc.from_list(compress(processors=post))
        self._batch_proc = Proc.from_list(compress(processors=batch))

    def with_(self, pre: ProcList = ..., vocab: ProcList = ..., post: ProcList = ..., batch: ProcList = ...) -> 'Pipe':
        self._pre_proc = Proc.from_list(subs(processors=pre, repl=self._pre_proc))
        self._vocab_proc = Proc.from_list(subs(processors=vocab, repl=self._vocab_proc))
        self._post_proc = Proc.from_list(subs(processors=post, repl=self._post_proc))
        self._batch_proc = Proc.from_list(subs(processors=batch, repl=self._batch_proc))
        return self

    def extra_repr(self):
        return ',\n  '.join([
            f'pre={self._pre_proc}',
            f'vocab={self._vocab_proc}',
            f'post={self._post_proc}',
            f'batch={self._batch_proc}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n  {self.extra_repr()}\n)'

    def preprocess(self, *datasets, name: str) -> Counter:
        counter = Counter()
        for dataset in datasets:
            todo = f'@{name}_{self.preprocess.__name__}_todo'
            if getattr(dataset, todo, True) and not isinstance(self._pre_proc, Identity):
                dataset.data[name] = [
                    self._pre_proc(datum, counter=counter, name=name)
                    for datum in dataset.data[name]
                ]
            setattr(dataset, todo, False)

        return counter

    def postprocess(self, *datasets, name: str) -> 'Pipe':
        _ = self.preprocess(*datasets, name=name)
        for dataset in datasets:
            todo = f'@{name}_{self.postprocess.__name__}_todo'
            if getattr(dataset, todo, True) and not isinstance(self._post_proc, Identity):
                dataset.data[name] = [
                    self._post_proc(datum, vocab=self.vocab, name=name)
                    for datum in dataset.data[name]
                ]
            setattr(dataset, todo, False)

        return self

    def build_vocab(self, *datasets, name: str = None,
                    special_tokens: Tuple[str, ...] = (),
                    max_size: Optional[int] = None, min_freq: int = 1) -> 'Pipe':
        counter = self.preprocess(*datasets, name=name)
        vocab = self._vocab_proc(
            counter, name=name,
            special_tokens=special_tokens,
            max_size=max_size, min_freq=min_freq,
        )
        assert isinstance(vocab, Vocab), f'{type(vocab)} is not Vocab'

        self.vocab = vocab
        return self

    def __call__(self, data: List[Any], name: str = '__call__') -> Tuple[Any, Vocab]:
        counter = Counter()
        data = [self._pre_proc(datum, counter=counter, name=name) for datum in data]
        vocab = self._vocab_proc(
            counter, name=name,
            special_tokens=(),
            max_size=None, min_freq=1,
        )
        batch = [self._post_proc(datum, vocab=vocab, name=name) for datum in data]
        return self._batch_proc(batch, vocab=vocab, name=name), vocab

    def collate_fn(self, batch: List[Any]) -> Any:
        return self._batch_proc(batch, vocab=self.vocab)


class RawPipe(Pipe):
    def __init__(self) -> None:
        super(RawPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=None,
        )

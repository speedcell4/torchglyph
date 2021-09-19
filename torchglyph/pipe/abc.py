from abc import ABCMeta
from collections import Counter
from typing import Optional, Union, List, Any, Tuple

from torchglyph.proc import Proc, Processors, compress, subs, Identity
from torchglyph.vocab import Vocab

__all__ = [
    'Pipe', 'RawPipe',
]


class Pipe(object, metaclass=ABCMeta):
    def __init__(self, pre: Processors = None, vocab: Processors = None,
                 post: Processors = None, batch: Processors = None) -> None:
        super(Pipe, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self.pre_proc = Proc.from_list(compress(processors=pre))
        self.vocab_proc = Proc.from_list(compress(processors=vocab))
        self.post_proc = Proc.from_list(compress(processors=post))
        self.batch_proc = Proc.from_list(compress(processors=batch))

    def with_(self, pre: Processors = ..., vocab: Processors = ...,
              post: Processors = ..., batch: Processors = ...) -> 'Pipe':
        self.pre_proc = Proc.from_list(subs(processors=pre, repl=self.pre_proc))
        self.vocab_proc = Proc.from_list(subs(processors=vocab, repl=self.vocab_proc))
        self.post_proc = Proc.from_list(subs(processors=post, repl=self.post_proc))
        self.batch_proc = Proc.from_list(subs(processors=batch, repl=self.batch_proc))
        return self

    def extra_repr(self) -> str:
        return ',\n  '.join([
            f'pre={self.pre_proc}',
            f'vocab={self.vocab_proc}',
            f'post={self.post_proc}',
            f'batch={self.batch_proc}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n  {self.extra_repr()}\n)'

    def preprocess_(self, *datasets, counter: Optional[Counter] = None) -> Counter:
        if counter is None:
            counter = Counter()

        if not isinstance(self.pre_proc, Identity):
            for dataset in datasets:
                for name, pipe in dataset.pipes.items():
                    if self is pipe:
                        todo = f'{name}_pre_todo'
                        if getattr(dataset, todo, True):
                            dataset.data[name] = [
                                self.pre_proc(datum, counter=counter, name=name)
                                for datum in dataset.data[name]
                            ]
                            setattr(dataset, todo, False)

        return counter

    def postprocess_(self, *datasets) -> 'Pipe':
        _ = self.preprocess_(*datasets)

        if not isinstance(self.post_proc, Identity):
            for dataset in datasets:
                for name, pipe in dataset.pipes.items():
                    if self is pipe:
                        todo = f'{name}_post_todo'
                        if getattr(dataset, todo, True):
                            dataset.data[name] = [
                                self.post_proc(datum, vocab=self.vocab, name=name)
                                for datum in dataset.data[name]
                            ]
                            setattr(dataset, todo, False)

        return self

    def build_vocab_(self, *datasets, special_tokens: Tuple[str, ...] = (),
                     max_size: Optional[int] = None, min_freq: int = 1) -> 'Pipe':
        name = ', '.join(sorted(list(set([
            name for dataset in datasets
            for name, pipe in dataset.pipes.items() if self is pipe
        ]))))

        self.vocab = self.vocab_proc(
            self.preprocess_(*datasets),
            name=f'[{name}]' if ', ' in name else name,
            special_tokens=special_tokens,
            max_size=max_size, min_freq=min_freq,
        )

        return self

    def __call__(self, data: List[Any], name: str = '__call__') -> Tuple[Any, Vocab]:
        counter = Counter()
        data = [self.pre_proc(datum, counter=counter, name=name) for datum in data]
        vocab = self.vocab_proc(counter, name=name, special_tokens=(), max_size=None, min_freq=1)
        batch = [self.post_proc(datum, vocab=vocab, name=name) for datum in data]
        return self.batch_proc(batch, vocab=vocab, name=name), vocab

    def collate_fn(self, batch: List[Any]) -> Any:
        return self.batch_proc(batch, vocab=self.vocab)


class RawPipe(Pipe):
    def __init__(self) -> None:
        super(RawPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=None,
        )

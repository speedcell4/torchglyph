from collections import Counter
from typing import Optional, Union, List, Any

from torchglyph.proc import Proc, PaLP, compress, subs
from torchglyph.vocab import Vocab


class Pipe(object):
    def __init__(self, pre: PaLP = None, vocab: PaLP = None, post: PaLP = None, batch: PaLP = None) -> None:
        super(Pipe, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self.pre = Proc.from_list(compress(pre))
        self.vocab = Proc.from_list(compress(vocab))
        self.post = Proc.from_list(compress(post))
        self.batch = Proc.from_list(compress(batch))

    def new_(self, pre: PaLP = ..., vocab: PaLP = ..., post: PaLP = ..., batch: PaLP = ...) -> 'Pipe':
        self.pre = Proc.from_list(subs(procs=pre, repl=self.pre))
        self.vocab = Proc.from_list(subs(procs=vocab, repl=self.vocab))
        self.post = Proc.from_list(subs(procs=post, repl=self.post))
        self.batch = Proc.from_list(subs(procs=batch, repl=self.batch))
        return self

    def extra_repr(self):
        return ',\n  '.join([
            f'pre={self.pre}',
            f'vocab={self.vocab}',
            f'post={self.post}',
            f'batch={self.batch}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(\n  {self.extra_repr()}\n)'

    def preprocess(self, *datasets) -> Counter:
        counter = Counter()
        for dataset in datasets:
            for key, pipe in dataset.pipes.items():
                flag = f'@{key}_{self.preprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.data[key] = [
                        self.pre(ins, counter=counter)
                        for ins in dataset.data[key]
                    ]
                    setattr(dataset, flag, True)

        return counter

    def postprocess(self, *datasets) -> 'Pipe':
        _ = self.preprocess(*datasets)
        for dataset in datasets:
            for name, pipe in dataset.pipes.items():
                flag = f'@{name}_{self.postprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.data[name] = [
                        self.post(ins, vocab=self.vocab)
                        for ins in dataset.data[name]
                    ]
                    setattr(dataset, flag, True)

        return self

    def build_vocab(self, *datasets) -> 'Pipe':
        counter = self.preprocess(*datasets)
        vocab = self.vocab(counter)
        if isinstance(vocab, Vocab):
            self.vocab = vocab
        else:
            raise ValueError(f"vocabulary building produced '{type(vocab).__name__}', "
                             f"instead of '{Vocab.__name__}'")
        return self

    def collate_fn(self, collected_ins: List[Any]) -> Any:
        return self.batch(collected_ins, vocab=self.vocab)

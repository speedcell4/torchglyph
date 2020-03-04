from collections import Counter
from typing import Optional, Union, Any, List, Dict

from torchglyph.vocab import Vocab


class Proc(object):
    def __add__(self, rhs: Optional[Union['Proc', 'Compose']]) -> Union['Proc', 'Compose']:
        if rhs is None:
            return self
        return Compose(self, rhs)

    def __radd__(self, lhs: Optional[Union['Proc', 'Compose']]) -> Union['Proc', 'Compose']:
        if lhs is None:
            return self
        return Compose(lhs, self)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return x


class Compose(Proc):
    def __init__(self, *procs: Optional[Union['Proc', 'Compose']]) -> None:
        super(Compose, self).__init__()
        self.procs = []
        for proc in procs:
            if proc is None:
                pass
            elif isinstance(proc, Proc):
                self.procs.append(proc)
            elif isinstance(proc, Compose):
                self.procs.extend(proc.procs)
            raise NotImplementedError(f'unsupported type :: {type(proc)}')

    def __add__(self, rhs: Optional[Union['Proc', 'Compose']]) -> 'Compose':
        return Compose(*self.procs, rhs)

    def __radd__(self, lhs: Optional[Union['Proc', 'Compose']]) -> 'Compose':
        return Compose(lhs, *self.procs)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        for process in self.procs:
            x = process(x, *args, **kwargs)
        return x


# recursive processes

class RecurStrProc(Proc):
    def process(self, data: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, str):
            return self.process(data, *args, **kwargs)
        return type(data)(self(datum, *args, **kwargs) for datum in data)


class RecurListStrProc(Proc):
    def process(self, data: List[str], *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, list):
            assert len(data) > 0
            if isinstance(data[0], str):
                return self.process(data, *args, **kwargs)
        return type(data)(self(datum, *args, **kwargs) for datum in data)


# stage processes

class PreProc(Proc):
    def __call__(self, ins: Any, counter: Counter) -> Any:
        raise NotImplementedError


class VocabProc(Proc):
    def __call__(self, vocab: Union[Counter, Vocab]) -> Union[Counter, Vocab]:
        raise NotImplementedError


class PostProc(Proc):
    def __call__(self, ins: Any, vocab: Vocab) -> Any:
        raise NotImplementedError


class BatchProc(Proc):
    def __call__(self, batch: List[Any], vocab: Vocab) -> Any:
        raise NotImplementedError


# pipeline

class Pipeline(object):
    def __init__(self,
                 pre_procs: Optional[Union[Compose, PreProc]],
                 vocab_procs: Optional[Union[Compose, VocabProc]],
                 post_procs: Optional[Union[Compose, PostProc]],
                 batch_procs: Optional[Union[Compose, BatchProc]]) -> None:
        super(Pipeline, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self._pre_processing = Compose(pre_procs)
        self._vocab_processing = Compose(vocab_procs)
        self._post_processing = Compose(post_procs)
        self._batch_processing = Compose(batch_procs)

    def preprocess(self, *datasets) -> Counter:
        counter = Counter()
        for dataset in datasets:
            for key, pipe in dataset.pipelines.items():
                flag = f'@{key}_{self.preprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.instances[key] = [
                        self._pre_processing(ins, counter=counter)
                        for ins in dataset.instances[key]
                    ]
                    setattr(dataset, flag, True)

        return counter

    def postprocess(self, *datasets) -> 'Pipeline':
        _ = self.preprocess(*datasets)
        for dataset in datasets:
            for name, pipe in dataset.pipelines.items():
                flag = f'@{name}_{self.postprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.instances[name] = [
                        self._post_processing(ins, vocab=self.vocab)
                        for ins in dataset.instances[name]
                    ]
                    setattr(dataset, flag, True)

        return self

    def build_vocab(self, *datasets) -> 'Pipeline':
        counter = self.preprocess(*datasets)
        self.vocab = self._vocab_processing(counter)
        return self

    def collate_fn(self, batch: Dict[str, List[Any]]) -> Any:
        return self._batch_processing(batch, vocab=self.vocab)

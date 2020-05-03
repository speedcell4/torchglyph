# Welcome to TorchGlyph

Data Processor Combinators for Natural Language Processing

[![Actions Status](https://github.com/speedcell4/torchglyph/workflows/unit-tests/badge.svg)](https://github.com/speedcell4/torchglyph/actions)

## Installation

Simply run this command in your terminal,

```bash
pip install torchglyph
```

## Quickstart

The atomic data processor of TorchGlyph is called `Proc`. Compose operator `+` is provided to produce complex `Proc` by composing two simple `Proc`s. 

```python
ToLower() + ReplaceDigits(repl_token='<digits>')
```

Composed `Proc`s act like data `Pipe`lines, where raw textual data is processed incrementally. According to the stages, they are roughly categorized into four-groups:

+ `pre` for processing *before* building vocabulary;
+ `vocab` for building and updating *vocabulary*;
+ `post` for precessing *after* building vocabulary;
+ `batch` for collating examples to build *batches*.

Defining the `Pipe`s of your dataset is the first step to build a dataset, you can build it from scratch, 

```python
class PackedIdxSeqPipe(Pipe):
    def __init__(self, device, dtype = torch.long) -> None:
        super(PackedIdxSeqPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PackSeq(enforce_sorted=False) + ToDevice(device=device),
        )
```

or you can simply manipulate existing `Pipe`s by calling `.with_` method.

```python
class PackedTokSeqPipe(PackedIdxSeqPipe):
    def __init__(self, device, unk_token, special_tokens = (), 
                 threshold = THRESHOLD, dtype = torch.long) -> None:
        super(PackedTokSeqPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, 
                           special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )
```
from torchglyph.pipe import Pipe


class RawStrPipe(Pipe):
    def __init__(self) -> None:
        super(RawStrPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=None,
        )

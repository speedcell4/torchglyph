from aku import Aku

from torchglyph.env import uuid as glyph_uuid
from torchglyph.hooks import summary

aku = Aku()


@aku.register
def uuid():
    print(glyph_uuid(), end='')


aku.register(summary)

aku.run()

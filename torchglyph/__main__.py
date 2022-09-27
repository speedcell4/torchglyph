from aku import Aku

from torchglyph.env import uuid as glyph_uuid
from torchglyph.hooks import summary

aku = Aku()


@aku.option
def uuid():
    print(glyph_uuid(), end='')


aku.option(summary)

aku.run()

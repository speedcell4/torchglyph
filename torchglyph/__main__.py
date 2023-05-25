from aku import Aku

from torchglyph.summary import summary

aku = Aku()

aku.register(summary)

aku.run()

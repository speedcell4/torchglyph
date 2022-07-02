from aku import Aku

from benchmark.nn.rnn import benchmark_lstm

aku = Aku()

aku.option(benchmark_lstm)

aku.run()

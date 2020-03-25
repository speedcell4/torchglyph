import logging
from typing import Union

from allennlp.data.dataset import Batch as AllenBatch
from allennlp.modules import Elmo as AllenELMo
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from torchglyph import data_path
from torchglyph.io import download_and_unzip


class ELMo(AllenELMo):
    root = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
    name = {
        'small': '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_',
        'medium': '2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_',
        'original': '2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_',
        '5.5B': '2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_',
    }

    def __init__(self, *, options_file, weight_file, pack_output, **kwargs):
        logging.info(f'loading pretrained {self.__class__.__name__} from {weight_file}')

        super(ELMo, self).__init__(
            options_file=options_file, weight_file=weight_file, **kwargs,
        )

        self.pack_output = pack_output
        self.embedding_dim = self.get_output_dim()

    @classmethod
    def from_pretrained(cls, weight: str, pack_output: bool = True,
                        num_output_representations: int = 2,
                        dropout: float = 0., freeze: bool = True) -> 'ELMo':
        elmo_path = data_path / cls.__name__.lower()
        options_file = download_and_unzip(
            url=cls.root + (cls.name[weight] + 'options.json'),
            dest=elmo_path / (cls.name[weight] + 'options.json'),
        )
        weight_file = download_and_unzip(
            url=cls.root + (cls.name[weight] + 'weights.hdf5'),
            dest=elmo_path / (cls.name[weight] + 'weights.hdf5'),
        )
        return cls(
            options_file=str(options_file), weight_file=str(weight_file),
            num_output_representations=num_output_representations,
            requires_grad=not freeze, dropout=dropout, pack_output=pack_output,
        )

    def extra_repr(self) -> str:
        args = [
            f'{self._elmo_lstm._elmo_lstm.input_size}',
            f'{self._elmo_lstm._elmo_lstm.hidden_size}',
            f'num_layers={self._elmo_lstm.num_layers}',
            f'dropout={self._dropout.p}',
        ]
        if not self._elmo_lstm._requires_grad:
            args.append('frozen')
        return ', '.join(args)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def forward(self, batch: AllenBatch) -> Union[Tensor, PackedSequence]:
        outputs = super(ELMo, self).forward(batch)
        elmo_representations, *_ = outputs['elmo_representations']
        if not self.pack_output:
            return elmo_representations
        else:
            lengths = outputs['mask'].long().sum(dim=-1)
            return pack_padded_sequence(
                elmo_representations, lengths,
                batch_first=True, enforce_sorted=False,
            )

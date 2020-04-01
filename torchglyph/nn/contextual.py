import json
import logging
from pathlib import Path
from typing import List
from typing import Union

from allennlp.modules import Elmo as AllenELMo
from elmoformanylangs.elmo import read_list, create_batches, recover
from elmoformanylangs.frontend import Model
from elmoformanylangs.modules.embedding_layer import EmbeddingLayer
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch.nn.utils.rnn import pack_sequence

from torchglyph import data_path
from torchglyph.io import download_and_unzip

logging.getLogger('elmoformanylangs').disabled = True


class ELMoModel(AllenELMo):
    root = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/'
    name = {
        'small': '2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_',
        'medium': '2x2048_256_2048cnn_1xhighway/elmo_2x2048_256_2048cnn_1xhighway_',
        'original': '2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_',
        '5.5B': '2x4096_512_2048cnn_2xhighway_5.5B/elmo_2x4096_512_2048cnn_2xhighway_5.5B_',
    }

    def __init__(self, *, options_file: str, weight_file: str, pack_output, **kwargs) -> None:
        logging.info(f'loading pretrained {self.__class__.__name__} from {weight_file}')

        super(ELMoModel, self).__init__(
            options_file=options_file, weight_file=weight_file, **kwargs,
        )

        self.pack_output = pack_output
        self.embedding_dim = self.get_output_dim()

    @classmethod
    def from_pretrained(cls, weight: str, pack_output: bool = True,
                        num_output_representations: int = 2,
                        dropout: float = 0., freeze: bool = True) -> 'ELMoModel':
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

    def forward(self, batch: Tensor, word_inputs: Tensor = None) -> Union[Tensor, PackedSequence]:
        outputs = super(ELMoModel, self).forward(batch, word_inputs=word_inputs)
        elmo_representations, *_ = outputs['elmo_representations']
        if not self.pack_output:
            return elmo_representations
        else:
            lengths = outputs['mask'].long().sum(dim=-1)
            return pack_padded_sequence(
                elmo_representations, lengths,
                batch_first=True, enforce_sorted=False,
            )


class ELMoForManyLanguages(Model):
    root = 'http://vectors.nlpl.eu/repository/11/'
    configs = [
        'https://raw.githubusercontent.com/HIT-SCIR/ELMoForManyLangs/master/configs/cnn_0_100_512_4096_sample.json',
        'https://raw.githubusercontent.com/HIT-SCIR/ELMoForManyLangs/master/configs/cnn_50_100_512_4096_sample.json',
    ]
    names = {
        'ca': '138',
        'es': '145',
        'zh': '179',
    }

    def __init__(self, *, options_file: Path, weight_file: Path, pack_output) -> None:
        with options_file.open('r', encoding='utf-8') as fp:
            config = json.load(fp)

        if config['token_embedder']['char_dim'] > 0:
            char_lexicon = {}
            with (weight_file / 'char.dic').open('r', encoding='utf-8') as fp:
                for raw in fp:
                    tokens = raw.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, index = tokens
                    char_lexicon[token] = int(index)
            char_emb_layer = EmbeddingLayer(
                config['token_embedder']['char_dim'], char_lexicon,
                fix_emb=False, embs=None,
            )
        else:
            char_lexicon = None
            char_emb_layer = None

        if config['token_embedder']['word_dim'] > 0:
            word_lexicon = {}
            with (weight_file / 'word.dic').open('r', encoding='utf-8') as fp:
                for raw in fp:
                    tokens = raw.strip().split('\t')
                    if len(tokens) == 1:
                        tokens.insert(0, '\u3000')
                    token, index = tokens
                    word_lexicon[token] = int(index)
            word_emb_layer = EmbeddingLayer(
                config['token_embedder']['word_dim'], word_lexicon,
                fix_emb=False, embs=None,
            )
        else:
            word_lexicon = None
            word_emb_layer = None

        super(ELMoForManyLanguages, self).__init__(
            config=config, word_emb_layer=word_emb_layer,
            char_emb_layer=char_emb_layer, use_cuda=False,
        )
        self.load_model(path=weight_file)
        self.char_lexicon = char_lexicon
        self.word_lexicon = word_lexicon
        self.pack_output = pack_output
        self.encoding_dim = self.output_dim * 2

    @classmethod
    def from_pretraiend(cls, lang: str, pack_output: bool = True) -> 'ELMoForManyLanguages':
        download_and_unzip(
            url=cls.configs[0],
            dest=data_path / cls.__name__.lower() / 'configs' / Path(cls.configs[0]).name,
        )
        download_and_unzip(
            url=cls.configs[1],
            dest=data_path / cls.__name__.lower() / 'configs' / Path(cls.configs[1]).name,
        )
        path = download_and_unzip(
            url=cls.root + f'{cls.names[lang]}.zip',
            dest=data_path / cls.__name__.lower() / lang / f'{lang}.zip',
        ).parent

        with (path / 'config.json').open('r', encoding='utf-8') as fp:
            args = json.load(fp)
        return cls(
            options_file=path / args['config_path'],
            weight_file=path, pack_output=pack_output,
        )

    def forward(self, batch: List[List[str]], output_layer: int = -1) -> Union[Tensor, PackedSequence]:
        if self.config['token_embedder']['name'].lower() == 'cnn':
            pad, text = read_list(batch, self.config['token_embedder']['max_characters_per_token'])
        else:
            pad, text = read_list(batch)

        pad_w, pad_c, pad_ln, pad_mask, pad_text, recover_idx = create_batches(
            pad, len(text), self.word_lexicon, self.char_lexicon, self.config, text=text)

        ans = []
        for word, char, length, mask, pads in zip(pad_w, pad_c, pad_ln, pad_mask, pad_text):
            output = super(ELMoForManyLanguages, self).forward(word, char, mask)
            for index, text in enumerate(pads):
                if self.config['encoder']['name'].lower() == 'lstm':
                    data = output[index, 1:length[index] - 1, :]
                elif self.config['encoder']['name'].lower() == 'elmo':
                    data = output[:, index, 1:length[index] - 1, :]

                if output_layer == -1:
                    payload = data.mean(dim=0)
                else:
                    payload = data[output_layer]
                ans.append(payload)

        ans = recover(ans, recover_idx)
        if self.pack_output:
            ans = pack_sequence(ans, enforce_sorted=False)
        return ans

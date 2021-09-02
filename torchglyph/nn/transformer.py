from torch import nn


class TransformerFfn(nn.Sequential):
    def __init__(self, dropout: float, bias: bool = True, *, in_features: int) -> None:
        super(TransformerFfn, self).__init__(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, in_features, bias=bias),
            nn.Tanh(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, in_features, bias=bias),
        )

        self.in_features = in_features
        self.dropout = dropout
        self.bias = bias

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'dropout={self.dropout}',
            f'bias={self.bias}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

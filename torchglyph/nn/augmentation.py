from typing import Tuple

import torch
from torch import nn
from torchrua import Z


class Aug(nn.Module):
    def __init__(self, p: float = 0.1, *, mask_token_id: int, special_token_ids: Tuple[int, ...]) -> None:
        super(Aug, self).__init__()

        self.p = p
        self.mask_token_id = mask_token_id

        special_token_ids = torch.tensor(special_token_ids, dtype=torch.long)
        self.register_buffer('special_token_ids', special_token_ids)

        boundaries = torch.tensor([p], dtype=torch.float32)
        self.register_buffer('boundaries', boundaries)

    def extra_repr(self) -> str:
        return f', '.join([
            f'p={self.p}',
            f'mask={self.mask_token_id}',
            f'specials={self.special_token_ids.detach().cpu().tolist()}',
        ])

    def forward(self, sequence: Z, **kwargs) -> Z:
        raise NotImplementedError


class TokenAug(Aug):
    def forward(self, sequence: Z, **kwargs) -> Z:
        if not self.training:
            return sequence

        data = sequence.data.clone()

        bucket = torch.rand_like(data, dtype=torch.float32)
        bucket = torch.bucketize(bucket, boundaries=self.boundaries)
        bucket[(data[..., None] == self.special_token_ids).any(dim=-1)] = -1

        data[bucket == 0] = self.mask_token_id

        return sequence._replace(data=data)

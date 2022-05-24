import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from torchrua import major_sizes_to_ptr


class VarDropout(nn.Dropout):
    def __init__(self, p: float) -> None:
        super(VarDropout, self).__init__(p=p, inplace=True)

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        if not self.training:
            return sequence

        mask = torch.ones(
            (sequence.batch_sizes[0].item(), *sequence.data.size()[1:]),
            dtype=sequence.data.dtype, device=sequence.data.device, requires_grad=False,
        )
        mask = super(VarDropout, self).forward(mask)

        batch_sizes = sequence.batch_sizes.to(device=sequence.data.device)
        batch_ptr, _ = major_sizes_to_ptr(sizes=batch_sizes)
        return sequence._replace(data=sequence.data * mask[batch_ptr])

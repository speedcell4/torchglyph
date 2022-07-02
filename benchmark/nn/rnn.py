from typing import Type

import torch
from torch import nn
from torchrua import pack_sequence
from tqdm import tqdm

from benchmark.generator import device
from torchglyph.meter import TimeMeter
from torchglyph.nn.rnn import Lstm


def benchmark_lstm(num_runs: int = 120, batch_size: int = 64, max_token_size: int = 120,
                   input_size: int = 512, hidden_size: int = 512,
                   bias: bool = True, bidirectional: bool = True,
                   meter: Type[TimeMeter] = TimeMeter):
    actual_lstm = Lstm(
        num_conjugates=1,
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias, bidirectional=bidirectional,
    ).to(device=device)

    excepted_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias, bidirectional=bidirectional,
    ).to(device=device)

    actual_forward = meter()
    actual_backward = meter()

    excepted_forward = meter()
    excepted_backward = meter()

    for _ in tqdm(range(num_runs)):
        token_sizes = torch.randint(1, max_token_size, (batch_size,), device=device).detach().cpu().tolist()

        actual_sequence = pack_sequence([
            torch.randn((token_size, 1, input_size), requires_grad=True, device=device)
            for token_size in token_sizes
        ], device=device)

        excepted_sequence = pack_sequence([
            torch.randn((token_size, input_size), requires_grad=True, device=device)
            for token_size in token_sizes
        ], device=device)

        with actual_forward:
            actual, _, _, _ = actual_lstm.forward(actual_sequence)

        with actual_backward:
            _ = torch.autograd.grad(
                actual, actual_sequence.data, torch.randn_like(actual),
                create_graph=False, only_inputs=True, allow_unused=False,
            )

        with excepted_forward:
            (excepted, _, _, _), _ = excepted_lstm.forward(excepted_sequence)

        with excepted_backward:
            _ = torch.autograd.grad(
                excepted, excepted_sequence.data, torch.randn_like(excepted),
                create_graph=False, only_inputs=True, allow_unused=False,
            )

        assert actual[:, 0, :].size() == excepted.size(), f'{actual.size()} == {excepted.size()}'

    print(f'TorchGlyph => {actual_forward.seconds_per_unit} + {actual_backward.seconds_per_unit}')
    print(f'PyTorch    => {excepted_forward.seconds_per_unit} + {excepted_backward.seconds_per_unit}')

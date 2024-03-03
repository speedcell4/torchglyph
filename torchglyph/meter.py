from logging import getLogger
from numbers import Number
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torchmetrics import Metric, MetricCollection

from torchglyph.dist import is_master
from torchglyph.serde import save_sota

logger = getLogger(__name__)


def detach(obj, *, n: int = 2):
    if isinstance(obj, float):
        return round(obj, ndigits=n)

    if torch.is_tensor(obj):
        return detach(obj.detach().cpu().item(), n=n)

    if isinstance(obj, (str, bool, int, complex)):
        return obj

    if isinstance(obj, (set, list, tuple)):
        return type(obj)(detach(o, n=n) for o in obj)

    if isinstance(obj, dict):
        return {k: detach(v, n=n) for k, v in obj.items()}

    raise TypeError(f'{type(obj)} is not supported')


class Meter(nn.Module):
    def stats(self) -> Tuple[Union[Number, Tensor], ...]:
        raise NotImplementedError

    def __gt__(self, other: 'Meter') -> bool:
        return detach(self.stats()) > detach(other.stats())

    def __lt__(self, other: 'Meter') -> bool:
        return detach(self.stats()) < detach(other.stats())

    def __eq__(self, other: 'Meter') -> bool:
        return detach(self.stats()) == detach(other.stats())

    def __ge__(self, other: 'Meter') -> bool:
        return detach(self.stats()) >= detach(other.stats())

    def __le__(self, other: 'Meter') -> bool:
        return detach(self.stats()) <= detach(other.stats())

    def __ne__(self, other: 'Meter') -> bool:
        return detach(self.stats()) != detach(other.stats())

    def reset(self) -> None:
        for metric in self._modules.values():
            if isinstance(metric, (Metric, MetricCollection)):
                metric.reset()

    def log(self, desc: str, step: int, out_dir: Path = None) -> None:
        for name, metric in self._modules.items():
            if isinstance(metric, (Metric, MetricCollection)):
                values = detach(metric.compute())

                if not isinstance(values, dict):
                    values = {name: values}
                else:
                    values = {f'{name}-{k}': v for k, v in values.items()}

                msg = ' | '.join(f'{key} {value}' for key, value in values.items())
                logger.info(f'{desc} {step} => {msg}')

                if out_dir is not None and is_master():
                    save_sota(out_dir=out_dir, step=step, **{
                        f'{desc}-{key}': value
                        for key, value in values.items()
                    })

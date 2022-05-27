from torch import Tensor


def linear(tensor: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = (weight @ tensor[..., None]).flatten(start_dim=-2)
    if bias is not None:
        out = out + bias
    return out

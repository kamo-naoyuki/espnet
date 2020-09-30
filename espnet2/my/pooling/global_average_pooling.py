import torch
from typeguard import check_argument_types

from espnet2.my.pooling.abs_pooling import AbsPooling


class GlobalAveragePooling(AbsPooling):
    def __init__(self, dim: int = 1):
        assert check_argument_types()
        super().__init__()
        self.dim = dim

    def extr_repr(self):
        return f"dim={self.dim}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """Forward

        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, D)
        """
        if x_lengths is None:
            return x.mean(dim=self.dim)
        else:
            if isinstance(self.dim, int):
                n = 1
            else:
                n = len(self.dim)
            # Assume the first dimension is the batch-axis
            axes = (slice(None),) + (None,) * (x.dim() - 1 - n)
            return x.sum(dim=self.dim) / x_lengths[axes].type(x.dtype)

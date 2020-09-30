import torch
from typeguard import check_argument_types

from espnet2.my.pooling.abs_pooling import AbsPooling


class GlobalMaxPooling(AbsPooling):
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
        # x_lengths is not used
        return x.max(dim=self.dim)[0]

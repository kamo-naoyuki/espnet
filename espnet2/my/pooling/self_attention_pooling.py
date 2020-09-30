"""
Adapted from https://github.com/clovaai/voxceleb_trainer/blob/master/accuracy.py
(MIT License)
"""
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.my.pooling.abs_pooling import AbsPooling


class SelfAttentionPooling(AbsPooling):
    def __init__(self, input_size: int, hidden_size: int = 512):
        super().__init__()
        assert check_argument_types()
        self.linear = torch.nn.Linear(input_size, hidden_size)
        self.attention = self.new_parameter(hidden_size, 1)

    @staticmethod
    def new_parameter(*size):
        out = torch.nn.Parameter(torch.FloatTensor(*size))
        torch.nn.init.xavier_normal_(out)
        return out

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """Forward

        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, D)
        """
        # x: (B, L, O) -> h: (B, L, O2)
        h = torch.tanh(self.linear(x))
        # h: (B, L, O2) -> w: (B, L)
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        if x_lengths is not None:
            w.masked_fill_(make_pad_mask(x_lengths, w, 1), -float("inf"))
        # w: (B, L) -> (B, L, 1)
        w = torch.nn.functional.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        # x: (B, L, O) * w: (B, L, 1) -> x: (B, L, O) -> x: (B, O)
        x = torch.sum(x * w, dim=1)
        return x

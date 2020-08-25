from typing import Tuple

import torch
from torch.nn import Parameter
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface


class ESPnetInstanceNorm1d(AbsNormalize, InversibleInterface):
    """InstanceNorm1d supporting zero-padded tensor

    The difference from InstanceNorm1d

    1. Normalize along the 2nd axis. i.e. Deal the input tensor as (B, D, C)
       where, B is batch size, D is feature size, C is channel number.
    2. Support variable length dimension using "input_lengths" tesnor

    >>> norm = ESPnetInstanceNorm1d(20)
    >>> bs, max_length, dim = 2, 10,  40
    >>> inp = torch.randn(2, max_length, dim)
    >>> inp_lengths = torch.LongTensor([10, 8])
    >>> out, _ = norm(inp, inp_lengths)

    3. Inversible

    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

    def extra_repr(self) -> str:
        repr_ = (
            f"num_features={self.num_features}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )

        if self.track_running_stats:
            repr_ += f", momentum={self.num_features}"
        repr_ += f", eps={self.eps}"
        return repr_

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, D)
            ilens: (B,)
        """
        if self.training and self.track_running_stats:
            self.num_batches_tracked = self.num_batches_tracked + 1

        if ilens is not None:
            # Fill 0
            mask = make_pad_mask(ilens, x, 1)
            x = x.masked_fill(mask, 0.0)
        else:
            mask = None

        if not self.training and self.track_running_stats:
            x = (x - self.running_mean[None, None, :]) / (
                self.running_var[None, None, :].sqrt() + self.eps
            )
        else:
            # Calc mean/var along length axis
            if ilens is not None:
                axes = [slice(None)] + [None for _ in range(x.dim() - 2)]
                mean = x.sum(dim=1) / ilens[axes]
                var = (x ** 2).sum(dim=1) / ilens[axes] - mean ** 2
            else:
                mean = x.mean(dim=1)
                var = x.var(dim=1, unbiased=False)

            # Update running mean and variance
            if self.track_running_stats:
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * mean.mean(0)
                self.running_var = (
                    1 - self.momentum
                ) * self.running_var + self.momentum * var.mean(0)

            x = (x - mean[:, None, :]) / (var[:, None, :].sqrt() + self.eps)
        if self.affine:
            x = self.weight[None, None, :] * x + self.bias[None, None, :]

        if ilens is not None:
            x = x.masked_fill(mask, 0.0)
        return x, ilens

    def inverse(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("inverse() can't be used if training mode")
        if not self.track_running_stats:
            raise RuntimeError("inverse() can't be used if track_running_stats False")
        if self.affine:
            x = (x - self.bias[None, None, :]) / (self.weight[None, None, :] + self.eps)
        x = (
            x * (self.running_var[None, None, :].sqrt() + self.eps)
            + self.running_mean[None, None, :]
        )
        if ilens is not None:
            x = x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)
        return x, ilens

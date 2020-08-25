"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)
"""

from typing import Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class AMSoftmaxLoss(AbsLoss):
    def __init__(
        self, input_size: int, num_classes: int, m: float = 0.3, s: float = 15.0
    ):
        assert check_argument_types()
        super().__init__()
        self.m = m
        self.s = s
        self.input_size = input_size
        self.W = torch.nn.Parameter(
            torch.randn(input_size, num_classes), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, x.size(-1))
        label = label.view(-1)

        assert x.size(0) == label.size(0), (x.size(), label.size)
        assert x.size(1) == self.input_size, x.size()

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        delt_costh = torch.zeros(costh.size(), device=x.device).scatter_(
            1, label_view, self.m
        )
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        loss = self.ce(costh_m_s, label)
        (prec1,) = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))
        return loss, prec1

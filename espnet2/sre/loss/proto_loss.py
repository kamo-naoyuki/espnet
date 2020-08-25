"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
Re-implementation of prototypical networks (https://arxiv.org/abs/1703.05175).
Numerically checked against https://github.com/cyvius96/prototypical-network-pytorch
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class ProtoLoss(AbsLoss):
    def __init__(self):
        assert check_argument_types()
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        output = -1 * (
            F.pairwise_distance(
                out_positive.unsqueeze(-1).expand(-1, -1, stepsize),
                out_anchor.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2),
            )
            ** 2
        )
        label = torch.arange(0, stepsize, device=x.device)
        nloss = self.criterion(output, label)
        (prec1,) = accuracy(output.detach(), label.detach(), topk=(1,))

        return nloss, prec1

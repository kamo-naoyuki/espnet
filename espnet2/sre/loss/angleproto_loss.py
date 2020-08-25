"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class AngleProtoLoss(AbsLoss):
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        assert check_argument_types()
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out_anchor = torch.mean(x[:, 1:, :], 1)
        out_positive = x[:, 0, :]
        stepsize = out_anchor.size()[0]

        cos_sim_matrix = F.cosine_similarity(
            out_positive.unsqueeze(-1).expand(-1, -1, stepsize),
            out_anchor.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2),
        )
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.arange(0, stepsize, device=x.device)
        nloss = self.criterion(cos_sim_matrix, label)
        (prec1,) = accuracy(cos_sim_matrix.detach(), label.detach(), topk=(1,))

        return nloss, prec1

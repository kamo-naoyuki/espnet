"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
Fast re-implementation of the GE2E loss (https://arxiv.org/abs/1710.10467)
Numerically checked against https://github.com/cvqluu/GE2E-Loss
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class GE2ELoss(AbsLoss):
    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        assert check_argument_types()
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        gsize = x.size()[1]
        centroids = torch.mean(x, 1)
        stepsize = x.size()[0]

        cos_sim_matrix = []

        for ii in range(0, gsize):
            idx = [*range(0, gsize)]
            idx.remove(ii)
            exc_centroids = torch.mean(x[:, idx, :], 1)
            cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
            cos_sim = F.cosine_similarity(
                x[:, ii, :].unsqueeze(-1).expand(-1, -1, stepsize),
                centroids.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2),
            )
            cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
            cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

        cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b

        label = torch.arange(0, stepsize, device=x.device)
        nloss = self.criterion(
            cos_sim_matrix.view(-1, stepsize),
            torch.repeat_interleave(label, repeats=gsize, dim=0),
        )
        (prec1,) = accuracy(
            cos_sim_matrix.view(-1, stepsize).detach(),
            torch.repeat_interleave(label, repeats=gsize, dim=0).detach(),
            topk=(1,),
        )

        return nloss, prec1

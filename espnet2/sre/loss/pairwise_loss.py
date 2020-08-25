"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
import random
from typing import Tuple

import numpy
import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.utils import calculate_eer


class PairwiseLoss(AbsLoss):
    def __init__(
        self,
        loss_func: str = "contrastive",
        hard_rank: int = 0,
        hard_prob: float = 0.0,
        margin: float = 0.0,
    ):
        assert check_argument_types()
        if loss_func not in ("contrastive", "triplet"):
            raise ValueError(f"loss_func={loss_func}")

        super().__init__()
        self.loss_func = loss_func
        self.hard_rank = hard_rank
        self.hard_prob = hard_prob
        self.margin = margin

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        out_anchor = x[:, 0, :]
        out_positive = x[:, 1, :]
        stepsize = out_anchor.size()[0]

        output = -1 * (
            F.pairwise_distance(
                out_anchor.unsqueeze(-1).expand(-1, -1, stepsize),
                out_positive.unsqueeze(-1).expand(-1, -1, stepsize).transpose(0, 2),
            )
            ** 2
        )

        negidx = self._mine_hard_negative(output.detach())

        out_negative = out_positive[negidx, :]

        # calculate distances
        pos_dist = F.pairwise_distance(out_anchor, out_positive)
        neg_dist = F.pairwise_distance(out_anchor, out_negative)

        # loss functions
        if self.loss_func == "contrastive":
            nloss = torch.mean(
                torch.cat(
                    [
                        torch.pow(pos_dist, 2),
                        torch.pow(F.relu(self.margin - neg_dist), 2),
                    ],
                    dim=0,
                )
            )
        elif self.loss_func == "triplet":
            nloss = torch.mean(
                F.relu(torch.pow(pos_dist, 2) - torch.pow(neg_dist, 2) + self.margin)
            )
        else:
            raise RuntimeError(f"loss_func={self.loss_func}")

        scores = -1 * torch.cat([pos_dist, neg_dist], dim=0).detach().cpu().numpy()
        label = numpy.array([1] * len(out_positive) + [0] * len(out_negative))
        eer, *_ = calculate_eer(label, scores)
        return nloss, torch.tensor(eer)

    def _mine_hard_negative(self, output):
        """Hard negative mining."""

        negidx = []

        for idx, similarity in enumerate(output):

            simval, simidx = torch.sort(similarity, descending=True)

            # Semi hard negative mining
            if self.hard_rank < 0:
                semihardidx = simidx[
                    (similarity[idx] - self.margin < simval)
                    & (simval < similarity[idx])
                ]

                if len(semihardidx) == 0:
                    negidx.append(random.choice(simidx))
                else:
                    negidx.append(random.choice(semihardidx))

            # Rank based negative mining
            else:
                simidx = simidx[simidx != idx]

                if random.random() < self.hard_prob:
                    negidx.append(simidx[random.randint(0, self.hard_rank)])
                else:
                    negidx.append(random.choice(simidx))

        return negidx

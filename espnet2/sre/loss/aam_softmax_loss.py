"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
Adapted from https://github.com/wujiyang/Face_Pytorch (Apache License)
"""
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class AAMSoftmaxLoss(AbsLoss):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        m: float = 0.3,
        s: float = 15.0,
        easy_margin: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self.m = m
        self.s = s
        self.input_size = input_size
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_classes, input_size), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(-1, x.size(-1))
        label = label.view(-1)
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.ce(output, label)
        (prec1,) = accuracy(output.detach(), label.detach(), topk=(1,))
        return loss, prec1

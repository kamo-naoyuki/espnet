"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.accuracy import accuracy


class SoftmaxLoss(AbsLoss):
    def __init__(self, input_size: int, num_classes: int):
        assert check_argument_types()
        super().__init__()

        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc = torch.nn.Linear(input_size, num_classes)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward.

        Args:
            x: (B, N, D), speaker vector
            label: (B, N,), speaker id label
        """
        # x: (B, N, D) -> x: (B, N, Nspk)
        x = self.fc(x)
        nloss = self.criterion(x.view(-1, x.size(-1)), label.view(-1))
        (prec1,) = accuracy(x.detach(), label.detach(), topk=(1,))

        return nloss, prec1

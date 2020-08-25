"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.sre.net.abs_net import AbsNet


class VGGVox(AbsNet):
    def __init__(self, input_size: int, output_size: int = 512):
        assert check_argument_types()
        if input_size != 40:
            raise NotImplementedError("input_size must be 40")
        super().__init__()

        self._input_size = input_size
        self._output_size = output_size
        self.netcnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 96, kernel_size=(5, 7), stride=(1, 2), padding=(2, 2)),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2)),
            torch.nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.Conv2d(256, 384, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.Conv2d(256, output_size, kernel_size=(4, 1), padding=(0, 0)),
            torch.nn.BatchNorm2d(output_size),
            torch.nn.ReLU(inplace=True),
        )

    def output_size(self):
        return self._output_size

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, L, O)
        """
        assert x.size(2) == self._input_size, (x.size(), self._input_size)

        # x: (B, L, D) -> (B, D, L) -> (B, 1, D, L)
        x = x.transpose(1, 2).unsqueeze(1)
        # x: (B, 1, D, L) -> (B, O, 1, L)
        x = self.netcnn(x)
        assert x.size(2) == 1, x.size()
        # x: (B, O, 1, L) -> (B, O, L)
        x = x.squeeze(dim=2)
        # x: (B, O, L) -> (B, L, O)
        x = x.transpose(1, 2)
        if x_lengths is not None:
            x = x.masked_fill(make_pad_mask(x_lengths, x, 1), 0.0)
        return x, x_lengths

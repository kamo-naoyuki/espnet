"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet2.sre.net.abs_net import AbsNet


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ResNet34(AbsNet):
    block = BasicBlock

    def __init__(
        self,
        input_size: int,
        layers: Sequence[int] = (3, 4, 6, 3),
        num_filters: Sequence[int] = (16, 32, 64, 128),
    ):
        assert check_argument_types()
        if input_size != 40:
            raise NotImplementedError("input_size must be 40")
        super().__init__()

        self._output_size = num_filters[-1]
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(
            1, num_filters[0], kernel_size=7, stride=(2, 1), padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(num_filters[0], layers[0])
        self.layer2 = self._make_layer(num_filters[1], layers[1], stride=(2, 2))
        self.layer3 = self._make_layer(num_filters[2], layers[2], stride=(2, 2))
        self.layer4 = self._make_layer(num_filters[3], layers[3], stride=(1, 1))

        self.avgpool = nn.AvgPool2d((5, 1), stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * self.block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.block.expansion),
            )

        layers = [self.block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * self.block.expansion
        for i in range(1, blocks):
            layers.append(self.block(self.inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def new_parameter(*size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out

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
        # x: (B, L, D) -> (B, D, L) -> (B, 1, D, L)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        # x_lengths -> x_lengths // 2
        x = self.layer2(x)
        # x_lengths -> x_lengths // 2
        x = self.layer3(x)
        x = self.layer4(x)

        # Average pooling along dimension axis
        x = self.avgpool(x)

        assert x.size(2) == 1, x.size()
        # x: (B, O, 1, L) -> (B, O, L)
        x = x.squeeze(dim=2)
        # x: (B, O, L) -> (B, L, O)
        x = x.transpose(1, 2)

        if x_lengths is not None:
            x_lengths = x_lengths // 2 // 2
            x = x.masked_fill(make_pad_mask(x_lengths, x, 1), 0.0)
        return x, x_lengths

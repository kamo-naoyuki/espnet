import torch

from espnet2.sre.net.resnet34 import ResNet34


def test_ResNet34():
    net = ResNet34(40)
    x = torch.randn(2, 100, 40)
    x_lengths = torch.tensor([100, 89], dtype=torch.long)
    y, _ = net(x, x_lengths)
    assert y.size(2) == net.output_size()
    y.sum().backward()

import torch

from espnet2.sre.pooling.global_max_pooling import GlobalMaxPooling


def test_GlobalMaxPooling():
    net = GlobalMaxPooling()
    x = torch.randn(2, 10, 40, requires_grad=True)
    x_lengths = torch.tensor([10, 8], dtype=torch.long)
    y = net(x, x_lengths)
    assert y.size() == (2, 40)
    y.sum().backward()

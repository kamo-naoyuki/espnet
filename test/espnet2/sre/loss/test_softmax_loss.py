import torch

from espnet2.sre.loss.softmax_loss import SoftmaxLoss


def test_SoftmaxLoss():
    loss = SoftmaxLoss(10, 10)
    x = torch.randn(2, 1, 10)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

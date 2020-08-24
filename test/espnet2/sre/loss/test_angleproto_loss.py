import torch

from espnet2.sre.loss.angleproto_loss import AngleProtoLoss


def test_AngleProtoLoss():
    loss = AngleProtoLoss()
    x = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

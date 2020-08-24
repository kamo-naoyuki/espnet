import torch

from espnet2.sre.loss.proto_loss import ProtoLoss


def test_ProtoLoss():
    loss = ProtoLoss()
    x = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

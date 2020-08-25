import torch

from espnet2.sre.loss.ge2e_loss import GE2ELoss


def test_Loss():
    loss = GE2ELoss()
    x = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

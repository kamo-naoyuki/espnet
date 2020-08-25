import torch

from espnet2.sre.loss.am_softmax_loss import AMSoftmaxLoss


def test_AMSoftmaxLoss():
    loss = AMSoftmaxLoss(10, 10)
    x = torch.randn(2, 1, 10)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

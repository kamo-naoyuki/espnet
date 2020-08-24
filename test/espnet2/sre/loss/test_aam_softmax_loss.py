import pytest
import torch

from espnet2.sre.loss.aam_softmax_loss import AAMSoftmaxLoss


@pytest.mark.parametrize("easy_mergin", [True, False])
def test_AAMSoftmaxLoss(easy_mergin):
    loss = AAMSoftmaxLoss(10, 10, easy_margin=easy_mergin)
    x = torch.randn(2, 1, 10)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

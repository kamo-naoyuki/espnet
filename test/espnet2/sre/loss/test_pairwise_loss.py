import pytest
import torch

from espnet2.sre.loss.pairwise_loss import PairwiseLoss


@pytest.mark.parametrize("loss_func", ["contrastive", "triplet"])
@pytest.mark.parametrize("hard_rank", [-1, 0, 1])
def test_PairwiseLoss(loss_func, hard_rank):
    loss = PairwiseLoss(loss_func=loss_func, hard_rank=hard_rank)
    x = torch.randn(2, 4, 10, requires_grad=True)
    t = torch.tensor([[0], [1]])
    l, _ = loss(x, t)
    l.backward()

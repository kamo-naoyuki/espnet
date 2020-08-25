import torch

from espnet2.sre.pooling.self_attention_pooling import SelfAttentionPooling


def test_SelfAttentionPooling():
    net = SelfAttentionPooling(40)
    x = torch.randn(2, 10, 40, requires_grad=True)
    x_lengths = torch.tensor([10, 8], dtype=torch.long)
    y = net(x, x_lengths)
    assert y.size() == (2, 40)
    y.sum().backward()

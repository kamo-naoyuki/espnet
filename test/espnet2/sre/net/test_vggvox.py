import torch

from espnet2.sre.net.vggvox import VGGVox


def test_VGGvox():
    net = VGGVox(40, output_size=10)
    x = torch.randn(2, 100, 40)
    x_lengths = torch.tensor([100, 89], dtype=torch.long)
    y, _ = net(x, x_lengths)
    assert y.size(2) == net.output_size()
    y.sum().backward()

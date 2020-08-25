import numpy as np
import pytest
import torch

from espnet2.layers.instance_norm import ESPnetInstanceNorm1d


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_repr(affine, track_running_stats):
    layer = ESPnetInstanceNorm1d(
        10, affine=affine, track_running_stats=track_running_stats
    )
    print(layer)


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_backward(affine, track_running_stats):
    layer = ESPnetInstanceNorm1d(
        10, affine=affine, track_running_stats=track_running_stats
    )
    x = torch.randn(1, 2, 10, requires_grad=True)
    y, _ = layer(x)
    y.sum().backward()


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_inverse_on_training_mode(affine, track_running_stats):
    layer = ESPnetInstanceNorm1d(
        10, affine=affine, track_running_stats=track_running_stats
    )
    layer.train()
    x = torch.randn(2, 3, 10)
    with pytest.raises(RuntimeError):
        y, _ = layer.inverse(x)


@pytest.mark.parametrize(
    "affine, track_running_stats",
    [(True, True), (False, False), (True, False), (False, True)],
)
def test_inverse_identity(affine, track_running_stats):
    layer = ESPnetInstanceNorm1d(
        10, affine=affine, track_running_stats=track_running_stats, eps=0.0
    )
    if affine:
        opt = torch.optim.Adam(layer.parameters())
    for _ in range(10):
        x = torch.randn(2, 3, 10)
        x_lens = torch.tensor([3, 2])
        y, _ = layer(x, x_lens)
        if affine:
            y.sum().backward()
            opt.step()

    layer.eval()
    x = torch.randn(2, 3, 10)
    y, _ = layer(x)

    with torch.no_grad():
        if not track_running_stats:
            with pytest.raises(RuntimeError):
                x2, _ = layer.inverse(y)
        else:
            x2, _ = layer.inverse(y)
            np.testing.assert_allclose(x.numpy(), x2.numpy(), rtol=1e-3)

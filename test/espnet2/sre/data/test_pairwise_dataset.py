import numpy
import pytest

from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.sre.data.pairwise_dataset import PairwiseDataset


@pytest.fixture
def feats_and_target(tmp_path):
    s = tmp_path / "npy.scp"
    w = NpyScpWriter(tmp_path / "data", s)
    w["1"] = x = numpy.random.randn(2, 40).astype(numpy.float)
    w["2"] = x2 = numpy.random.randn(2, 40).astype(numpy.float)
    w["3"] = x3 = numpy.random.randn(2, 40).astype(numpy.float)
    w["4"] = x4 = numpy.random.randn(2, 40).astype(numpy.float)
    return s, x, x2, x3, x4


def test_PairwiseDataset_repr(feats_and_target):
    feats = feats_and_target[0]
    dataset = PairwiseDataset(path_name_type_list=[(str(feats), "feats", "npy")])
    print(dataset)


def test_PairwiseDataset_name(feats_and_target):
    feats = feats_and_target[0]
    dataset = PairwiseDataset(path_name_type_list=[(str(feats), "feats", "npy")])
    print(dataset.names())
    assert dataset.has_name("feats")


def test_PairwiseDataset_getitem(feats_and_target):
    feats, x, x2, x3, x4 = feats_and_target
    dataset = PairwiseDataset(path_name_type_list=[(str(feats), "feats", "npy")])
    _, d = dataset[("1", "2")]
    _, d2 = dataset[("3", "4")]
    y, y2 = d["feats"]
    y3, y4 = d2["feats"]
    numpy.testing.assert_allclose(x, y)
    numpy.testing.assert_allclose(x2, y2)
    numpy.testing.assert_allclose(x3, y3)
    numpy.testing.assert_allclose(x4, y4)

import pytest

from espnet2.sre.utils import calculate_eer
from espnet2.sre.utils import read_utt2spk


@pytest.fixture
def utt2spk(tmp_path):
    key = tmp_path / "key_file"
    with key.open("w") as f:
        f.write("1 sp1\n")
        f.write("2 sp2\n")
        f.write("3 sp1\n")
        f.write("4 sp2\n")
        f.write("5 sp2\n")
        f.write("6 sp1\n")
        f.write("7 sp2\n")
        f.write("8 sp2\n")
        f.write("9 sp3\n")
        f.write("10 sp1\n")
    return key


@pytest.fixture
def utt2spk_three_columns(tmp_path):
    key = tmp_path / "key_file"
    with key.open("w") as f:
        f.write("1 sp1 1\n")
    return key


@pytest.fixture
def utt2spk_empty(tmp_path):
    key = tmp_path / "key_file"
    with key.open("w") as f:
        pass
    return key


@pytest.fixture
def utt2spk_duplicated(tmp_path):
    key = tmp_path / "key_file"
    with key.open("w") as f:
        f.write("1 sp1\n")
        f.write("1 sp2\n")
    return key


def test_read_utt2spk(utt2spk):
    utt2spk_dict, spk2utt_dict, spk2spkid, spkid2spk = read_utt2spk(utt2spk)
    assert len(utt2spk_dict) == 10
    assert len(spk2utt_dict) == 3
    for utt, spk in utt2spk_dict.items():
        assert spk in spk2utt_dict
        assert spk in spk2spkid
        assert spk2spkid[spk] in spkid2spk


def test_read_utt2spk_three_columns(utt2spk_three_columns):
    with pytest.raises(RuntimeError):
        read_utt2spk(utt2spk_three_columns)


def test_read_utt2spk_empty(utt2spk_empty):
    with pytest.raises(RuntimeError):
        read_utt2spk(utt2spk_empty)


def test_read_utt2spk_duplicated(utt2spk_duplicated):
    with pytest.raises(RuntimeError):
        read_utt2spk(utt2spk_duplicated)


def test_calculate_eer():
    calculate_eer([1, 0], [0.2, 0.5])

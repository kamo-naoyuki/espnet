import pytest

from espnet2.sre.data.pairwise_batch_sampler import PairwiseBatchSampler


@pytest.fixture
def utt2spk(tmp_path):
    key = tmp_path / "key_file"
    with key.open("w") as f:
        f.write("1 sp1\n")
        f.write("2 sp2\n")
        f.write("3 sp3\n")
        f.write("4 sp4\n")
        f.write("5 sp5\n")
        f.write("6 sp6\n")
        f.write("7 sp7\n")
        f.write("8 sp8\n")
        f.write("9 sp9\n")
        f.write("10 sp10\n")
    return key


@pytest.mark.parametrize("shuffle", [True, False])
def test_PairwiseBatchSampler_repr(utt2spk, shuffle):
    sampler = PairwiseBatchSampler(
        key_file=utt2spk, utt2spk=utt2spk, batch_size=2, num_pair=2, shuffle=shuffle,
    )
    print(sampler)


@pytest.mark.parametrize("shuffle", [True, False])
def test_PairwiseBatchSampler_len(utt2spk, shuffle):
    sampler = PairwiseBatchSampler(
        key_file=utt2spk, utt2spk=utt2spk, batch_size=2, num_pair=2, shuffle=shuffle,
    )
    assert len(sampler) == 5


@pytest.mark.parametrize("shuffle", [True, False])
def test_PairwiseBatchSampler_generate(utt2spk, shuffle):
    sampler = PairwiseBatchSampler(
        key_file=utt2spk, utt2spk=utt2spk, batch_size=2, num_pair=2, shuffle=shuffle,
    )
    for i in sampler.generate():
        assert len(i) == 2


@pytest.mark.parametrize("shuffle", [True, False])
def test_PairwiseBatchSampler_empty(utt2spk, shuffle, tmp_path):
    k = tmp_path / "k"
    with k.open("w") as f:
        pass
    with pytest.raises(RuntimeError):
        PairwiseBatchSampler(
            key_file=k, utt2spk=utt2spk, batch_size=2, num_pair=2, shuffle=shuffle,
        )


@pytest.mark.parametrize("shuffle", [True, False])
def test_PairwiseBatchSampler_too_large_num_pair(utt2spk, shuffle):
    with pytest.raises(RuntimeError):
        PairwiseBatchSampler(
            key_file=utt2spk,
            utt2spk=utt2spk,
            batch_size=2,
            num_pair=100,
            shuffle=shuffle,
        )

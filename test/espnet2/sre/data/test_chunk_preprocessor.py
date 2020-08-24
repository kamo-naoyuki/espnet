import numpy
import pytest

from espnet2.sre.data.chunk_preprocessor import FeatsExtractChunkPreprocessor


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


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_repr(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type=feats_type,
    )
    print(preprocessor)


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_call(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type=feats_type,
    )
    # Wav input
    x = numpy.random.randint(-100, 100, 16000)
    label = numpy.array([0])

    d = {"speech": x}
    preprocessor("1", d)

    d = {"speech": x, "reference": x, "label": label}
    preprocessor("1", d)

    # Feats input
    x = numpy.random.randint(-100, 100, 16000)
    d = {"speech": x}
    preprocessor("1", d)


def test_FeatsExtractChunkPreprocessor_unknown_feats_type(utt2spk):
    with pytest.raises(ValueError):
        FeatsExtractChunkPreprocessor(
            utt2spk=utt2spk, train=False, fs=16000, feats_type="aaa",
        )


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_unknown_keys(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type=feats_type,
    )
    x = numpy.random.randint(-100, 100, 16000)
    with pytest.raises(RuntimeError):
        preprocessor("1", {"aaa": x})


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_not_remained(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type=feats_type,
    )
    x = numpy.zeros((16000,), dtype=numpy.int64)
    with pytest.raises(RuntimeError):
        preprocessor("1", {"speech": x})


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_short_utt(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type=feats_type, chunk_length=100,
    )
    # Wav input
    x = numpy.random.randint(-100, 100, 1600)
    d = {"speech": x}
    with pytest.raises(RuntimeError):
        preprocessor("1", d)


def test_FeatsExtractChunkPreprocessor_negative_chunk_length(utt2spk):
    with pytest.raises(ValueError):
        FeatsExtractChunkPreprocessor(
            utt2spk=utt2spk, train=False, fs=16000, feats_type="mfcc", chunk_length=-1,
        )


@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("feats_type", ["spectrogram", "mfcc", "fbank"])
def test_FeatsExtractChunkPreprocessor_get_num_features(utt2spk, train, feats_type):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk,
        train=train,
        fs=16000,
        feats_type=feats_type,
        n_fft=512,
        n_mels=40,
        n_cep=20,
    )
    if feats_type == "spctrogram":
        assert preprocessor.get_num_features() == 257
    elif feats_type == "fbank":
        assert preprocessor.get_num_features() == 40
    else:
        assert preprocessor.get_num_features() == 20


@pytest.mark.parametrize("train", [True, False])
def test_FeatsExtractChunkPreprocessor_get_num_features(utt2spk, train):
    preprocessor = FeatsExtractChunkPreprocessor(
        utt2spk=utt2spk, train=train, fs=16000, feats_type="mfcc", n_mels=40,
    )
    assert preprocessor.get_num_spk() == 10

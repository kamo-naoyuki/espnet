import logging
from pathlib import Path
from typing import Dict
from typing import Optional
from typing import Union

import librosa
import numpy as np
from scipy.fftpack import dct
from typeguard import check_argument_types

from espnet2.sre.utils import read_utt2spk


def lifter(cepstra: np.ndarray, L: int = 22) -> np.ndarray:
    """Apply a cepstral lifter the the matrix of cepstra.

    Ref:
    https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/base.py
    (MIT License)

    Args:
        cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2.0) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def preemphasis(signal: np.ndarray, coeff=0.95) -> np.ndarray:
    """perform preemphasis on the input signal.

    Ref:
    https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
    (MIT License)

    Args:
        signal: The signal to filter.
        coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    returns:
        the filtered signal.
    """
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


class FeatsExtractChunkPreprocessor:
    def __init__(
        self,
        train: bool,
        utt2spk: Union[str, Path],
        fs: int,
        feats_type: str = "mfcc",
        preemp_coeff: float = 0.97,
        n_fft: int = 512,
        hop_length: int = 128,
        window: str = "hann",
        center: bool = True,
        win_length: int = None,
        pad_mode: str = "reflect",
        lifter_coeff: int = 22,
        n_mels: int = 40,
        n_cep: int = 40,
        fmin: Optional[int] = 20,
        fmax: int = -400,
        apply_vad: bool = True,
        vad_energy_threshold: float = 0.0,
        vad_energy_mean_scale: float = 0.5,
        cut_chunk: bool = True,
        chunk_length: int = 100,
        seed: int = 0,
        eps: float = 1.0e-10,
    ):
        assert check_argument_types()
        if cut_chunk and chunk_length <= 0:
            raise ValueError(f"chunk_length < 0: {chunk_length}")
        if feats_type not in ("spectrogram", "fbank", "mfcc"):
            raise ValueError(f"feats_type={feats_type}")

        self.train = train
        self.utt2spk_file = utt2spk
        self.utt2spk, _, self.spk2spkid, _ = read_utt2spk(utt2spk)
        self.feats_type = feats_type
        self.preemp_coeff = preemp_coeff

        # STFT related
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = window
        self.center = center
        if win_length is None:
            win_length = n_fft
        self.win_length = win_length
        self.pad_mode = pad_mode

        # MFCC related
        self.fs = fs
        self.fmin = 0 if fmin is None else fmin
        self.fmax = (
            self.fs / 2 if fmax is None else (fmax if fmax > 0 else self.fs / 2 - fmax)
        )
        self.lifter_coeff = lifter_coeff
        self.n_mels = n_mels
        self.n_cep = n_cep

        # VAD related
        self.apply_vad = apply_vad
        self.vad_energy_threshold = vad_energy_threshold
        self.vad_energy_mean_scale = vad_energy_mean_scale

        # Chunk related
        self.cut_chunk = cut_chunk
        self.chunk_length = chunk_length
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

        self.eps = eps

    def __repr__(self):
        _ret = f"{self.__class__.__name__}("
        items = list(self.__dict__.items())
        for k, v in items:
            if k == items[-1][0]:
                _ret += f"{k}={v})"
            else:
                _ret += f"{k}={v}, "
        return _ret

    def get_num_spk(self) -> int:
        return len(self.spk2spkid)

    def get_num_features(self) -> int:
        if self.feats_type == "mfcc":
            return self.n_cep
        elif self.feats_type == "fbank":
            return self.n_mels
        elif self.feats_type == "spectrogram":
            return self.n_fft // 2 + 1
        else:
            raise RuntimeError(f"feats_type={self.feats_type}")

    def __call__(self, uid: str, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if set(data) == {"speech"}:
            # 1. Get the spkid and set it
            spkid = self.spk2spkid[self.utt2spk[uid]]
            data["spkid"] = np.array(spkid)

            x = data["speech"]
            assert isinstance(x, np.ndarray), type(x)
            xs = [x]

        elif set(data) == {"speech", "reference", "label"}:
            x = data["speech"]
            y = data["reference"]
            assert isinstance(x, np.ndarray), type(x)
            assert isinstance(y, np.ndarray), type(y)
            if data["label"].ndim != 1:
                raise RuntimeError(f"Unexpected label shape: {data['label'].shape}")
            if len(data["label"]) != 1:
                raise RuntimeError(f"Label must be scalar: {data['label']}")

            xs = [x, y]

        else:
            raise RuntimeError(f"Unexpected keys: {set(data)}")

        retval = []
        for x in xs:
            if x.ndim > 2 or x.ndim == 0:
                raise RuntimeError(f"Unexpected speech shape: {x.shape}")

            # If x.ndim == 2, x is the extracted feature.
            elif x.ndim == 1:

                # 2.a. STFT
                x = preemphasis(x, self.preemp_coeff)
                # x: (Nsamples,) -> (Frames, Freq)
                x = librosa.stft(
                    x,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    window=self.window,
                    center=self.center,
                    pad_mode=self.pad_mode,
                ).T
                assert x.shape[1] == self.n_fft // 2 + 1
                power = x.real ** 2 + x.imag ** 2
                log_energy = np.log(np.maximum(self.eps, np.sum(power, axis=1)))

                if self.feats_type in ("fbank", "mfcc"):
                    # 2.b. log-FBANK
                    # mel: (Mel_freq, Freq)
                    mel = librosa.filters.mel(
                        self.fs, self.n_fft, self.n_mels, self.fmin, self.fmax
                    )
                    # log_power: (Frames, Mel_freq)
                    log_power = np.log(np.maximum(self.eps, np.dot(power, mel.T)))
                    x = log_power

                    if self.feats_type == "mfcc":
                        # 2.c. MFCC
                        cep = dct(log_power, type=2, axis=1, norm="ortho")[
                            :, : self.n_cep
                        ]
                        mfcc = lifter(cep, self.lifter_coeff)
                        x = mfcc

                elif self.feats_type != "spectrogram":
                    raise RuntimeError(f"feats_type={self.feats_type}")

                # 2.d. VAD
                if self.apply_vad:
                    """
                    Implementation of Energy based VAD in Kaldi
                    Ref:
                    https://github.com/kaldi-asr/kaldi/blob/master/src/ivector/voice-activity-detection.cc
                    
                    Not supporting context now
                    """
                    vad_energy_threshold = self.vad_energy_threshold
                    if self.vad_energy_mean_scale != 0.0:
                        vad_energy_threshold += (
                            self.vad_energy_mean_scale * log_energy.mean()
                        )
                    vad = x[log_energy > vad_energy_threshold]

                    logging.debug(f"VAD: {uid}: {len(x)}frames -> {len(vad)}frames")
                    x = vad
                    if len(vad) == 0:
                        raise RuntimeError(
                            f"0 frames remain after VAD. Reconsider VAD parameters: "
                            f"uid={uid}, "
                            f"log_energy=(mean: {log_energy.mean()}, "
                            f"max: {log_energy.max()}, "
                            f"min: {log_energy.min()}), "
                            f"vad_energy_threshold={self.vad_energy_threshold}, "
                            f"vad_energy_mean_scale={self.vad_energy_mean_scale}",
                        )

            # 3. Cut chunk
            if self.cut_chunk:
                if len(x) < self.chunk_length:
                    # Pad with wrap mode if utterance is too short
                    x = np.pad(
                        x, [(0, self.chunk_length - len(x)), (0, 0)], mode="wrap",
                    )

                elif len(x) == self.chunk_length:
                    pass

                else:
                    if self.train:
                        # Derive a chunk from utterance randomly
                        # Note(kamo): If num_workers>1, the result is not deterministic.
                        offset = self.random_state.randint(
                            0, len(x) - self.chunk_length
                        )
                        x = x[offset : offset + self.chunk_length]
                    else:
                        x = x[: self.chunk_length]

            retval.append(x)

        if len(retval) == 1:
            data["speech"] = retval[0]
        else:
            data["speech"] = retval[0]
            data["reference"] = retval[1]

        return data

from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet2.layers.stft import Stft
from espnet2.tts.feats_extract.abs_feats_extract import AbsFeatsExtract
from espnet2.utils.griffin_lim import Spectrogram2Waveform


class LogSpectrogram(AbsFeatsExtract):
    """Conventional frontend structure for ASR

    Stft -> log-amplitude-spec
    """

    def __init__(
        self,
        n_fft: int = 512,
        win_length: Union[int, None] = 512,
        hop_length: int = 128,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided
        )
        self.n_fft = n_fft

    def output_size(self) -> int:
        return self.n_fft // 2 + 1

    def build_griffin_lim_vocoder(self, griffin_lim_iters) -> Spectrogram2Waveform:
        return Spectrogram2Waveform(
            fs=16000,
            griffin_lim_iters=griffin_lim_iters,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            n_mels=None,
            win_length=self.win_length,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # STFT -> Power spectrum
        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        log_amp = 0.5 * torch.log(input_power + 1.e-20)
        return log_amp, feats_lens
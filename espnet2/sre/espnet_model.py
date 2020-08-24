from typing import Dict
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.net.abs_net import AbsNet
from espnet2.sre.pooling.abs_pooling import AbsPooling
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel


class ESPnetSREModel(AbsESPnetModel):
    def __init__(
        self,
        normalize: Optional[torch.nn.Module],
        net: AbsNet,
        pooling: AbsPooling,
        loss: AbsLoss,
        linear: torch.nn.Linear,
        apply_l2_normalize: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        self.normalize = normalize
        self.net = net
        self.pooling = pooling
        self.linear = linear
        self.loss = loss
        self.apply_l2_normalize = apply_l2_normalize

    def forward(
        self, **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Forward function. """
        # Training mode
        if "spkid" in kwargs:
            assert set(kwargs) == {"speech", "spkid"}, set(kwargs)
            return self._forward_train(**kwargs)

        # Evaluation mode
        else:
            assert set(kwargs) == {
                "speech",
                "speech_lengths",
                "reference",
                "reference_lengths",
                "label",
            }, set(kwargs)
            return self._forward_evaluate(**kwargs)

    def _forward_train(self, speech: torch.Tensor, spkid: torch.Tensor):
        """Training mode.

        Args (train mode):
            speech: (B, N, L, D), Extracted feature. e.g. MFCC
            spkid: (B, N,)

        """
        assert speech.dim() == 4, speech.shape
        assert spkid.dim() == 2, spkid.shape
        assert speech.size()[:2] == spkid.size()[:2], (speech.size(), spkid.size())

        x = speech
        size = x.size()
        # x: (B, N, L, D) -> (B * N, L, D)
        x = x.view(-1, size[2], size[3])
        # x: (B * N, L, D) -> (B * N, O)
        x = self.compute_embed_vector(x)
        # x: (B * N, O) -> (B, N, O)
        x = x.view(size[0], size[1], x.size(1))
        loss, acc = self.loss(x, spkid)

        stats = dict(loss=loss.detach(), acc=acc.detach() if acc is not None else None)
        batch_size = len(x)
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _forward_evaluate(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        reference: torch.Tensor,
        reference_lengths: torch.Tensor,
        label: torch.Tensor,
    ):
        """Evaluation mode.

        Args (eval mode):
            speech: (B, L, D), Extracted feature. e.g. MFCC
            speech_lengths: (B,)
            reference: (B, L, D), Extracted feature. e.g. MFCC
            reference_lengths: (B,)
            label: (B,) Label having 0 or 1 values
        """
        assert speech.dim() == 3, speech.shape
        assert speech_lengths.dim() == 1, speech_lengths.shape
        assert reference.dim() == 3, reference.shape
        assert reference_lengths.dim() == 1, reference_lengths.shape

        # dist: (B,)
        score, _, _ = self.compute_score(
            speech=speech,
            speech_lengths=speech_lengths,
            reference=reference,
            reference_lengths=reference_lengths,
        )

        # TODO(kamo): Implement EER calculation,
        #   but this is hard due to the system restriction of ESPnet2.
        score2 = (2 * label - 1) * score - label + 1
        score2 = score2.mean()

        stats = dict(score=score2.detach())
        batch_size = len(speech)
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((None, stats, batch_size), score2.device)
        return loss, stats, weight

    def compute_score(
        self,
        speech: torch.Tensor,
        reference: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        reference_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute score between two input speech

        Args (eval mode):
            speech: (B, L, D), Extracted feature. e.g. MFCC
            speech_lengths: (B,)
            reference: (B, L, D), Extracted feature. e.g. MFCC
            reference_lengths: (B,)
            label: (B,)
        Returns:
            score, x, y: (B,), (B, O), (B, O)
        """
        assert speech.dim() == 3, speech.shape
        if speech_lengths is not None:
            assert speech_lengths.dim() == 1, speech_lengths.shape
        assert reference.dim() == 3, reference.shape
        if reference_lengths is not None:
            assert reference_lengths.dim() == 1, reference_lengths.shape

        x = self.compute_embed_vector(speech, speech_lengths)
        y = self.compute_embed_vector(reference, reference_lengths)

        # x: (B, O), y: (B, O) -> dist: (B,)
        score = torch.nn.functional.pairwise_distance(x, y)
        return score, x, y

    def compute_embed_vector(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor = None,
    ):
        assert speech.dim() == 3, speech.shape
        if self.normalize is not None:
            if speech_lengths is not None:
                assert speech_lengths.dim() == 1, speech_lengths.shape
            speech, _ = self.normalize(speech, speech_lengths)

        # speech: (B, L, D) -> x: (B, L2, D2)
        x, x_lengths = self.net(speech, speech_lengths)
        assert x.dim() == 3, x.size()
        if x_lengths is not None:
            assert x.size(0) == x_lengths.size(0), (x.size(), x_lengths.size(0))
            assert x_lengths.dim() == 1, x_lengths.size()

        # x: (B, L2, D2) -> x: (B, D2)
        x = self.pooling(x, x_lengths)
        assert x.dim() == 2, x.size()
        assert x.size(0) == speech.size(0), x.size()

        # x: (B, D2) -> x: (B, D3)
        x = self.linear(x)

        if self.apply_l2_normalize:
            x = torch.nn.functional.normalize(x, p=2, dim=1)
        return x

    def collect_feats(self, **kwargs) -> Dict[str, torch.Tensor]:
        """collect_feats.

        Args:
            speech: (B, L, D) Note that B is always 1. Extracted feature. e.g. MFCC
            spkid: (B,)
        """
        # Training mode
        if "spkid" in kwargs:
            assert set(kwargs) == {"speech", "spkid"}, set(kwargs)
            speech = kwargs["speech"]
            spkid = kwargs["spkid"]

            assert speech.dim() == 3, speech.shape
            assert spkid.dim() == 1, spkid.shape
            return {
                "speech": speech,
                "speech_lengths": torch.full(
                    (speech.size(0),),
                    fill_value=speech.size(1),
                    dtype=torch.long,
                    device=speech.device,
                ),
            }

        # Evaluation mode
        else:
            assert set(kwargs) == {
                "speech",
                "speech_lengths",
                "reference",
                "reference_lengths",
                "label",
            }, set(kwargs)
            speech = kwargs["speech"]
            speech_lengths = kwargs["speech_lengths"]
            reference = kwargs["reference"]
            reference_lengths = kwargs["reference_lengths"]
            return {
                "speech": speech,
                "speech_lengths": speech_lengths,
                "reference": reference,
                "reference_lengths": reference_lengths,
            }

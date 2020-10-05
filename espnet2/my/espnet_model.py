from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.my.net.conv_net import ConvNet
from espnet2.my.pooling.self_attention_pooling import SelfAttentionPooling
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    pass
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ClassifyCriterion(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        hidden_size2: int = 256,
        num_classes: int = 2,
        pooling_type: str = "SAP",
    ):
        assert check_argument_types()
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)
        if pooling_type == "SAP":
            self.pooling = SelfAttentionPooling(hidden_size, hidden_size2)
        else:
            raise RuntimeError

    def forward(self, state: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        assert state.dim() == 3, state.size()
        assert label.dim() == 1, label.size()

        # (B, L, D) -> (B, L, D2)
        embed = self.fc1(state)
        # (B, L, D) -> (B, D)
        embed = self.pooling(embed)
        # (B, D) -> (B, num_class)
        embed = self.fc2(embed)
        loss = self.criterion(embed, label.long())
        return loss


class Combiner(torch.nn.Module):
    def __init__(
        self,
        embed_size: int,
        embed_size2: int,
        hidden_size: int = 256,
    ):
        assert check_argument_types()
        super().__init__()
        self.embed_size = embed_size
        self.embed_size2 = embed_size2
        self.fc1 = torch.nn.Linear(embed_size2, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, embed_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.size(2) == self.embed_size, (x.size(), self.embed_size)
        assert y.size(2) == self.embed_size2, (y.size(), self.embed_size2)
        h = torch.relu(self.fc1(y))
        h = self.fc2(h)
        return x * h


class SDRCriterion(torch.nn.Module):
    def __init__(self, eps: float = 1e-20):
        assert check_argument_types()
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, x.size()
        assert x.size() == y.size(), (x.size(), y.size())

        # minus SDR (= DSR)
        return (
            10
            * torch.log10(
                torch.clamp(
                    torch.norm(x - y, dim=1)
                    / torch.clamp(torch.norm(y, dim=1), min=self.eps),
                    min=self.eps,
                )
            ).mean(0)
        )


class NMSECriterion(torch.nn.Module):
    def __init__(self, eps: float = 1e-20):
        assert check_argument_types()
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (B, T, D)
            y:
        """
        assert x.dim() == 3, x.size()
        loss = torch.nn.functional.mse_loss(x, y, reduction="none")
        # (B, T, D) -> (B,)
        loss = loss.mean([1, 2])
        n = torch.norm(y, dim=2).mean(1)
        n = torch.clamp(n, min=self.eps)
        return (loss / n).mean(0)


class IdentityCriterion(torch.nn.Module):
    def __init__(self, embed_size: int = 256, eps: float = 1e-20):
        assert check_argument_types()
        super().__init__()
        self.code_book = torch.nn.Parameter(torch.ones(embed_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x: (B, T, D)
            y:
        """
        assert x.dim() == 3, x.size()
        y = self.code_book[None, None, :]
        x, y = torch.broadcast_tensors(x, y)
        loss = torch.nn.functional.mse_loss(x, y, reduction="none")
        # (B, T, D) -> (B,)
        loss = loss.mean([1, 2])
        n = torch.norm(self.code_book, dim=0, keepdim=True)
        n = torch.clamp(n, min=self.eps)
        return (loss / n).mean(0)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        output_size: int = 256,
        window_size: int = 20,
        window_shift: int = 10,
    ):
        assert check_argument_types()
        super().__init__()
        self.conv = torch.nn.Conv1d(
            1,
            output_size,
            kernel_size=window_size,
            stride=window_shift,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, x.size()
        x = x.view(x.size(0), 1, x.size(1))
        x = self.conv(x)
        return x.transpose(2, 1)


class Decoder(torch.nn.Module):
    def __init__(
        self,
        input_size: int = 256,
        window_size: int = 20,
        window_shift: int = 10,
    ):
        assert check_argument_types()
        super().__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            input_size,
            1,
            kernel_size=window_size,
            stride=window_shift,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, x.size()
        x = x.transpose(2, 1)
        x = self.deconv(x)
        assert x.size(1) == 1, x.size()
        return x.squeeze(1)


class ESPnetMyModel(AbsESPnetModel):
    def __init__(
        self,
        encode_size: int = 256,
        embed_size: int = 256,
        embed_size2: int = 256,
        hidden_size: int = 256,
        hidden_size2: int = 256,
        hidden_size3: int = 256,
    ):
        assert check_argument_types()
        super().__init__()
        self.embed_size = embed_size
        self.embed_size2 = embed_size2
        self.encoder = Encoder(output_size=encode_size)
        self.encode_net = ConvNet(
            input_size=encode_size, output_size=embed_size + embed_size2, R=2,
        )
        self.combiner = Combiner(
            embed_size=embed_size, embed_size2=embed_size2, hidden_size=hidden_size
        )

        self.decode_net = ConvNet(input_size=embed_size, output_size=encode_size, R=2)
        self.decoder = Decoder(input_size=encode_size)
        self.classify_criterion = ClassifyCriterion(
            input_size=embed_size2,
            hidden_size=hidden_size2,
            hidden_size2=hidden_size3,
        )
        self.reconstruct_criterion = SDRCriterion()
        self.nmse_criterion = NMSECriterion()
        self.identity_criterion = IdentityCriterion(embed_size2)

    def forward(
        self,
        speech: torch.Tensor,
        label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            label: (Batch,) 0 indicates clean, 1 indicates noisy.
        """
        assert speech.size(0) == label.size(0), (speech.shape, label.shape)
        assert label.dtype == torch.long, label.dtype
        if label.dim() == 2 and label.size(1) == 1:
            label = label.squeeze(1)
        elif label.dim() != 1:
            raise ValueError(f"Must be 2 dim tensor: {label.size()}")

        label = label.bool()

        # The first iteration
        h_clean, h_noise = self._encode(speech)
        h_noisy = self.combiner(h_clean, h_noise)
        decode_clean = self._decode(h_clean)
        decode_noisy = self._decode(h_noisy)

        loss_classify = self.classify_criterion(h_noise, label)
        loss_noisy = self.reconstruct_criterion(decode_noisy, speech)
        loss = loss_classify + loss_noisy

        stats = dict(
            loss_classify=loss_classify.detach(),
            dsr_noisy=loss_noisy.detach(),
        )

        if (~label).sum() != 0:
            loss_clean = self.reconstruct_criterion(decode_clean[~label], speech[~label])
            loss_iden = self.identity_criterion(h_noise[~label])

            weight_clean = (~label).sum().float() / len(speech)
            loss += weight_clean * (loss_clean + loss_iden)
            stats.update(
                dsr_clean=loss_clean.detach(),
                loss_iden=loss_iden.detach(),
            )

        # The second iteration
        if label.sum() != 0 and loss_noisy < 0:
            speech2 = decode_clean[label]
            label2 = label[label]

            h2_clean, h2_noise = self._encode(speech2)
            h2_noisy = self.combiner(h2_clean, h2_noise)
            h1_2_noisy = self.combiner(h2_clean, h_noise[label])
            decode2_clean = self._decode(h2_clean)
            decode2_noisy = self._decode(h2_noisy)
            decode1_2_noisy = self._decode(h1_2_noisy)

            loss2_classify = self.classify_criterion(h2_noise, label2)
            loss2_clean = self.reconstruct_criterion(decode2_clean, speech2)
            loss2_noisy = self.reconstruct_criterion(decode2_noisy, speech2)
            loss1_2_noisy = self.reconstruct_criterion(decode1_2_noisy, speech[label])
            loss2_iden = self.identity_criterion(h2_noise)
            # loss2_hidden_clean = self.nmse_criterion(h2_clean, h_clean[label])

            weight_noisy = label.sum().float() / len(speech)
            loss += weight_noisy * (
                loss2_classify
                + 0.1 * loss2_clean
                + 0.1 * loss2_noisy
                + loss1_2_noisy
                + loss2_iden
            )

            stats.update(
                loss2_classify=loss2_classify.detach(),
                dsr2_clean=loss2_clean.detach(),
                dsr2_noisy=loss2_noisy.detach(),
                dsr1_2_noisy=loss1_2_noisy.detach(),
                loss2_iden=loss2_iden.detach(),
            )
        stats.update(loss=loss)

        batch_size = speech.size(0)
        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def inference(self, speech: torch.Tensor):
        h_clean, h_noise = self._encode(speech)
        h_noisy = self.combiner(h_clean, h_noise)
        decode_clean = self._decode(h_clean)
        decode_noisy = self._decode(h_noisy)
        return decode_clean,  decode_noisy

    def collect_feats(
        self,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _encode(self, speech: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # (B, T) -> (B, L, D)
        encoded = self.encoder(speech)
        # (B, L, D) -> (B, L, D2)
        h = self.encode_net(encoded)
        assert h.size(2) == self.embed_size + self.embed_size2, (
            h.size(),
            self.embed_size,
            self.embed_size2,
        )

        h_clean, h_noisy = h[:, :, : self.embed_size], h[:, :, self.embed_size :]
        return h_clean, h_noisy

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        # (B, L, D2) -> (B, L, D)
        encoded = self.decode_net(hidden)
        # (B, L, D) -> (B, T)
        speech = self.decoder(encoded)

        return speech


if __name__ == "__main__":
    net = ESPnetMyModel(2, 3, 4)
    x = torch.randn(3, 100)
    label = torch.tensor([0, 1, 1], dtype=torch.long)

    loss, stats, weight = net(x, label)
    print(stats)
    loss.backward()

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typeguard import check_argument_types


class ChannelWiseLayerNorm(nn.LayerNorm):
    def forward(self, x):
        # x: (B, C, T)
        if x.dim() != 3:
            raise RuntimeError("Expect (Batch, Channel, Time): {}".format(x.size()))
        # x: (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        # The statistics is gathered along time axis
        x = super().forward(x)
        # x: (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        return x


# TODO(kamo): Implement cumulative layer normalize for causal Tasnet
class GlobalChannelLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(torch.zeros(dim, 1))
            self.gamma = nn.Parameter(torch.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # x: (B, C, T)
        if x.dim() != 3:
            raise RuntimeError("Expect (Batch, Channel, Time): {}".format(x.size()))
        # mean: (B, 1, 1)
        mean = x.mean((1, 2), keepdim=True)
        # var: (B, 1, 1)
        var = ((x - mean) ** 2).mean((1, 2), keepdim=True)
        # B x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / torch.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return (
            "{normalized_dim}, eps={eps}, "
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)
        )


def build_norm(norm, dim):
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    elif norm == "gLN":
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)
    else:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))


class Conv1DBlock(nn.Module):
    """1D convolutional block:

    Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(
        self,
        in_channels=256,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
        norm="cLN",
        causal=False,
        padding: bool = True,
    ):
        super().__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        if padding:
            if not causal:
                dconv_pad = (dilation * (kernel_size - 1)) // 2
            else:
                dconv_pad = dilation * (kernel_size - 1)
        else:
            dconv_pad = 0

        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        y = x + y
        return y


class ConvNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        X: int = 8,
        R: int = 4,
        B: int = 256,
        H: int = 512,
        P: int = 3,
        norm: str = "cLN",
        num_srcs: int = 2,
        non_linear: str = "relu",
        causal: bool = False,
        domain: str = "time",
        n_input_channel: int = 1,
        normalize: bool = True,
    ):
        assert check_argument_types()
        if causal:
            warnings.warn("Norm layer is not causal now.")
        if domain == "tasnet":
            domain = "time"

        super().__init__()
        self.num_srcs = num_srcs
        self.non_linear_type = non_linear
        self.domain = domain
        self.n_input_channel = n_input_channel
        self.normalize = normalize

        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": torch.sigmoid,
            "softmax": F.softmax,
        }

        if non_linear not in supported_nonlinear:
            raise RuntimeError(
                "Unsupported non-linear function: {}", format(non_linear)
            )

        self.non_linear = supported_nonlinear[non_linear]

        if normalize:
            self.ln = ChannelWiseLayerNorm(input_size)
        else:
            self.ln = None
        self.conv1x1_1 = nn.Conv1d(input_size, B, 1)

        self.repeats = nn.Sequential(
            *[
                nn.Sequential(
                    *[
                        Conv1DBlock(
                            in_channels=B,
                            conv_channels=H,
                            kernel_size=P,
                            norm=norm,
                            causal=causal,
                            dilation=(2 ** b),
                        )
                        for b in range(X)
                    ]
                )
                for _ in range(R)
            ]
        )

        self.conv1x1_2 = nn.Conv1d(B, output_size, 1)

    def forward(self, speech: torch.Tensor):
        speech = speech.transpose(1, 2)
        # 2. Channel wise normalize
        if self.normalize:
            y = self.ln(speech)
        else:
            y = speech

        # 3. conv1x1, i.e. Linear
        # w: (Batch, N, Frame) -> y: (Batch, B,  Frame)
        y = self.conv1x1_1(y)

        # 4. Res blocks
        # y: (Batch, B,  Frame) -> (Batch, B,  Frame)
        y = self.repeats(y)

        # 5. conv1x1, i.e. Linear
        # y: (Batch, B, Frame) -> (Batch, Nsrc * N, Frame)
        y = self.conv1x1_2(y)
        y = y.transpose(1, 2)
        return y


if __name__ == "__main__":
    net = ConvNet(10, 20)
    x = torch.randn(2, 100, 10)

    y = net(
        x,
    )
    print(y.shape)

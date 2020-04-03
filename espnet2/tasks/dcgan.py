"""DCGAN example using ESPnet."""

import argparse
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.abs_task import optim_classes
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool


class Generator(torch.nn.Sequential):
    def __init__(self, z_size: int, num_channel: int, hidden_size: int = 64):
        super().__init__(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(z_size, hidden_size * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 8),
            torch.nn.ReLU(True),
            # state size. (hidden_size*8) x 4 x 4
            torch.nn.ConvTranspose2d(
                hidden_size * 8, hidden_size * 4, 4, 2, 1, bias=False
            ),
            torch.nn.BatchNorm2d(hidden_size * 4),
            torch.nn.ReLU(True),
            # state size. (hidden_size*4) x 8 x 8
            torch.nn.ConvTranspose2d(
                hidden_size * 4, hidden_size * 2, 4, 2, 1, bias=False
            ),
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.ReLU(True),
            # state size. (hidden_size*2) x 16 x 16
            torch.nn.ConvTranspose2d(hidden_size * 2, hidden_size, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size),
            torch.nn.ReLU(True),
            # state size. (hidden_size) x 32 x 32
            torch.nn.ConvTranspose2d(hidden_size, num_channel, 4, 2, 1, bias=False),
            torch.nn.Tanh()
            # state size. (num_channel) x 64 x 64
        )


class Discriminator(torch.nn.Sequential):
    def __init__(self, num_channel, hidden_size: int = 64):
        super().__init__(
            # input is (num_channel) x 64 x 64
            torch.nn.Conv2d(num_channel, hidden_size, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size) x 32 x 32
            torch.nn.Conv2d(hidden_size, hidden_size * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*2) x 16 x 16
            torch.nn.Conv2d(hidden_size * 2, hidden_size * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*4) x 8 x 8
            torch.nn.Conv2d(hidden_size * 4, hidden_size * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(hidden_size * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (hidden_size*8) x 4 x 4
            torch.nn.Conv2d(hidden_size * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid(),
        )


generator_choices = ClassChoices(
    "generator",
    classes=dict(generator1=Generator),
    type_check=torch.nn.Module,
    default="generator1",
)
discriminator_choices = ClassChoices(
    "discriminator",
    classes=dict(discriminator1=Discriminator),
    type_check=torch.nn.Module,
    default="discriminator1",
)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DCGANTrainer(Trainer):
    @classmethod
    def train_step(cls, options, optim_idx, model, batch, train_states):
        if optim_idx == 0:
            # Train discriminator
            loss, stats, weight = model(**batch)
        elif optim_idx == 1:
            # Train generator
            loss, stats, weight = model()
        else:
            raise RuntimeError("3 optimizers are not supported.")
        return loss, stats, weight, train_states


class ESPnetDCGANModel(AbsESPnetModel):
    def __init__(
        self, generator: torch.nn.Module, discriminator: torch.nn.Module, z_size: int,
    ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.z_size = z_size
        self.criterion = torch.nn.BCELoss()
        self.fake = None

    def collect_feats(self, image_0: torch.Tensor = None, image_1: torch.Tensor = None):
        return {"image_0": image_0}

    def forward(self, image_0: torch.Tensor = None, image_1: torch.Tensor = None):
        """Forward function.

        Args:
            image_0: Real image data
            image_1: The class label. This isn't used for this training.
        """
        real_label = 0
        fake_label = 1

        # Discriminator branch
        if image_0 is not None:
            B = image_0.size(0)
            label = torch.full((B,), real_label, device=image_0.device)

            # 1. Train with real
            output = self.discriminator(image_0)
            loss_D_real = self.criterion(output, label)

            # 2. Train with fake
            noise = torch.randn(B, self.z_size, 1, 1, device=image_0.device)

            # Set fake attribute. This is used for next forward()
            self.fake = self.generator(noise)
            label = torch.full((B,), fake_label, device=image_0.device)
            output = self.discriminator(self.fake.detach())
            loss_D_fake = self.criterion(output, label)

            loss = loss_D_real + loss_D_fake
            stats = dict(
                lodd_D=loss_D_real.detach() + loss_D_fake.detach(),
                loss_D_real=loss_D_real.detach(),
                loss_D_fake=loss_D_fake.detach(),
            )

            # For validation stage
            if not self.training:
                output = self.discriminator(self.fake)
                loss_G = self.criterion(output, label)
                stats.update(
                    loss=loss.detach() + loss_G.detach(), loss_G=loss_G.detach(),
                )
            else:
                stats.update(loss=loss.detach())

            loss, stats, B = force_gatherable((loss, stats, B), loss.device)
            return loss, stats, B

        # Generator branch
        elif self.fake is not None:
            B = self.fake.size(0)

            label = torch.full((B,), real_label, device=self.fake.device)
            label.fill_(real_label)  # fake labels are real for generator cost
            output = self.discriminator(self.fake)
            loss = self.criterion(output, label)
            stats = dict(loss_G=loss.detach())

            self.fake = None
            loss, stats, B = force_gatherable((loss, stats, B), loss.device)
            return loss, stats, B
        else:
            raise RuntimeError("One of image_0 or self.fake is required.")


class DCGANTask(AbsTask):
    num_optimizers: int = 2

    # Add variable objects configurations
    class_choices_list = [generator_choices, discriminator_choices]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = DCGANTrainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        assert check_argument_types()
        parser.set_defaults(num_att_plot=0, optim="adam")
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["num_channel"]

        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetDCGANModel),
            help="The keyword arguments for model class.",
        )
        group.add_argument(
            "--num_channel", type=int, help="The number of channels of input image"
        )
        group.add_argument(
            "--z_size", type=int, default=100, help="The size of the latent z vector"
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

        assert check_return_type(parser)
        return parser

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[
        [Collection[Tuple[str, Dict[str, np.ndarray]]]],
        Tuple[List[str], Dict[str, torch.Tensor]],
    ]:
        assert check_argument_types()
        return CommonCollateFn(not_sequence=["image_0", "image_1"])

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            raise NotImplementedError
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
        retval = ("image",)
        return retval

    @classmethod
    def optional_data_names(cls, inference: bool = False) -> Tuple[str, ...]:
        retval = ()
        return retval

    @classmethod
    def build_optimizers(
        cls, args: argparse.Namespace, model: torch.nn.Module,
    ) -> List[torch.optim.Optimizer]:
        optim_class = optim_classes.get(args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim}")
        optim = optim_class(model.discriminator.parameters(), **args.optim_conf)

        optim_class2 = optim_classes.get(args.optim2)
        if optim_class2 is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {args.optim2}")
        optim2 = optim_class2(model.generator.parameters(), **args.optim2_conf)
        return [optim, optim2]

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetDCGANModel:
        assert check_argument_types()
        # 1. Build generator and discriminator
        generator_class = generator_choices.get_class(args.generator)
        generator = generator_class(
            z_size=args.z_size, num_channel=args.num_channel, **args.generator_conf,
        )
        generator.apply(weights_init)

        discriminator_class = discriminator_choices.get_class(args.discriminator)
        discriminator = discriminator_class(
            num_channel=args.num_channel, **args.discriminator_conf,
        )
        discriminator.apply(weights_init)

        # 2. Build model
        model = ESPnetDCGANModel(
            generator=generator,
            discriminator=discriminator,
            z_size=args.z_size,
            **args.model_conf,
        )

        assert check_return_type(model)
        return model

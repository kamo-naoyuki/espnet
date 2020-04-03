#!/usr/bin/env python3

"""DCGAN decoding."""

import logging
from pathlib import Path
import sys
from typing import Optional
from typing import Union

import configargparse
import torch
import torchvision
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.dcgan import DCGANTask
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed


@torch.no_grad()
def inference(
    output_dir: str,
    dtype: str,
    ngpu: int,
    seed: int,
    log_level: Union[int, str],
    train_config: Optional[str],
    model_file: Optional[str],
    num_plots: int,
):
    """Perform DCGAN decoding."""
    assert check_argument_types()
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    set_all_random_seed(seed)

    # 2. Build model
    model, train_args = DCGANTask.build_model_from_file(
        train_config, model_file, device
    )
    model.to(dtype=getattr(torch, dtype)).eval()
    logging.info(f"Model :\n{model}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    to_PIL = torchvision.transforms.ToPILImage()
    # 3. Start for-loop
    for idx in range(num_plots):
        noise = torch.randn(1, model.z_size, 1, 1, device=device)
        fake = model.generator(noise)
        fake = to_PIL(fake[0])
        fake.save(output_dir / f"{idx}.png")


def get_parser():
    """Get argument parser."""
    parser = configargparse.ArgumentParser(
        description="Plot images generated by DCGAN model",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument("--config", is_config_file=True, help="config file path")

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu", type=int, default=0, help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_plots", type=int, default=10, help="The number of plots",
    )
    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    return parser


def main(cmd=None):
    """Run main func."""
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()

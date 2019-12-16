#!/usr/bin/env python3
import argparse
import logging
import random
import sys
import time
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import kaldiio
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.tts import TTSTask
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.dataset import ESPnetDataset
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def tts_decode(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: Optional[str],
    model_file: Optional[str],
    threshold: float,
    minlenratio: float,
    maxlenratio: float,
    use_att_constraint: bool,
    backward_window: int,
    forward_window: int,
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) "
        "%(levelname)s: %(message)s",
    )

    if ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    # 1. Set random-seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # 2. Build model
    with Path(train_config).open("r") as f:
        train_args = yaml.load(f, Loader=yaml.Loader)
    train_args = argparse.Namespace(**train_args)
    model = TTSTask.build_model(train_args)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device=device, dtype=getattr(torch, dtype)).eval()
    tts = model.tts
    normalize = model.normalize

    # 3. Build data-iterator
    dataset = ESPnetDataset(
        data_path_and_name_and_type,
        float_dtype=dtype,
        preprocess=TTSTask.build_preprocess_fn(train_args, False),
    )
    TTSTask.check_task_requirements(dataset, allow_variable_data_keys, False)
    if key_file is None:
        key_file, _, _ = data_path_and_name_and_type[0]

    batch_sampler = ConstantBatchSampler(
        batch_size=batch_size, key_file=key_file, shuffle=False
    )

    logging.info(f"Normalization:\n{normalize}")
    logging.info(f"TTS:\n{tts}")
    logging.info(f"Batch sampler: {batch_sampler}")
    logging.info(f"dataset:\n{dataset}")
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=TTSTask.build_collate_fn(train_args),
        num_workers=num_workers,
    )

    # 4. Start for-loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # FIXME(kamo): I think we shouldn't depend on kaldi-format any more.
    #  How about numpy or HDF5?
    #  >>> with NpyScpWriter() as f:
    with kaldiio.WriteHelper(
        "ark,scp:{o}.ark,{o}.scp".format(o=output_dir / "feats")
    ) as f:
        for idx, (keys, batch) in enumerate(loader, 1):
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            key = keys[0]
            # Change to single sequence and remove *_length
            # because inference() requires 1-seq, not mini-batch.
            _data = {
                k: v[0] for k, v in batch.items() if not k.endswith("_lengths")
            }
            start_time = time.perf_counter()

            # TODO(kamo): Use common gathering attention system and plot it
            #  Now att_ws is not used.

            with torch.no_grad():
                outs, probs, att_ws = tts.inference(
                    **_data,
                    threshold=threshold,
                    maxlenratio=maxlenratio,
                    minlenratio=minlenratio,
                )
                if normalize is not None:
                    outs, _ = normalize.inverse(outs[None])
                outs = outs.squeeze(0)

            insize = next(iter(_data.values())).size(0)
            logging.info(
                "inference speed = {} msec / frame.".format(
                    (time.perf_counter() - start_time)
                    / (int(outs.size(0)) * 1000)
                )
            )
            if outs.size(0) == insize * maxlenratio:
                logging.warning(
                    f"output length reaches maximum length ({key})."
                )

            logging.info(
                f"({idx}/{len(batch_sampler)}) {key} "
                f"(size:{insize}->{outs.size(0)})"
            )
            f[key] = outs.cpu().numpy()


def get_parser():
    parser = configargparse.ArgumentParser(
        description="TTS Decode",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--config", is_config_file=True, help="config file path"
    )

    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument(
        "--allow_variable_data_keys", type=str2bool, default=False
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument("--train_config", type=str)
    group.add_argument("--model_file", type=str)

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value in decoding",
    )
    group.add_argument(
        "--use_att_constraint",
        type=str2bool,
        default=False,
        help="Whether to use attention constraint",
    )
    group.add_argument(
        "--backward_window",
        type=int,
        default=1,
        help="Backward window value in attention constraint",
    )
    group.add_argument(
        "--forward_window",
        type=int,
        default=3,
        help="Forward window value in attention constraint",
    )
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    tts_decode(**kwargs)


if __name__ == "__main__":
    main()

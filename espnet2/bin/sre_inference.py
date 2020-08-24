#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.bin.calc_eer import write_eer
from espnet2.fileio.datadir_writer import DatadirWriter
from espnet2.fileio.npy_scp import NpyScpWriter
from espnet2.tasks.sre import SRETask
from espnet2.torch_utils.device_funcs import to_device
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


class SPeakerEmbedding:
    """Speaker class

    Examples:
        >>> import soundfile
        >>> embed = SPeakerEmbedding("asr_config.yml", "asr.pth")
        >>> audio, rate = soundfile.read("speech.wav")
        >>> assert rate == embed.train_args.fs
        >>> embed(audio)
        [(text, token, token_int, hypothesis object), ...]

    """

    def __init__(
        self,
        train_config: Union[Path, str],
        model_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
    ):
        assert check_argument_types()
        model, train_args = SRETask.build_model_from_file(
            train_config, model_file, device
        )
        model.to(dtype=getattr(torch, dtype)).eval()

        self.model = model
        self.train_args = train_args
        self.device = device
        self.dtype = dtype
        self.preprocess_fn = SRETask.build_preprocess_fn(train_args, False)

    @torch.no_grad()
    def __call__(
        self,
        speech: Union[torch.Tensor, np.ndarray],
        speech_lengths: Union[torch.Tensor, np.ndarray] = None,
        reference: Union[torch.Tensor, np.ndarray] = None,
        reference_lengths: Union[torch.Tensor, np.ndarray] = None,
    ):
        """Inference

        Args:
            speech: Input speech data
            speech_lengths:
            reference:
            reference_lengths:
        Returns:
            text, token, token_int, hyp

        """
        assert check_argument_types()

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        if isinstance(reference, np.ndarray):
            reference = torch.tensor(reference)

        # 1. Type check
        if reference_lengths is not None and reference is None:
            raise TypeError(
                "reference must be not None if reference_lengths is not None"
            )
        # Batch mode
        if speech_lengths is not None or reference_lengths is not None:
            batch_mode = True

            if speech_lengths is None:
                raise TypeError(
                    "speech_lengths must be not None if reference_lengths is not None"
                )
            if isinstance(speech_lengths, np.ndarray):
                speech_lengths = torch.tensor(speech_lengths)
            if speech.dim() < 2:
                raise ValueError(f"Expect (B, L) or (B, L, D): {speech.size()}")
            bs = speech.size(0)
            if speech_lengths.size(0) != bs:
                raise ValueError(
                    f"Must have same batch-size: {speech_lengths.size(0)} != {bs}"
                )

            if reference is not None:
                if reference_lengths is None:
                    raise TypeError("reference_lengths must be not None")
                if isinstance(reference_lengths, np.ndarray):
                    reference_lengths = torch.tensor(reference_lengths)
                if reference.dim() < 2:
                    raise ValueError(f"Expect (B, L) or (B, L, D): {reference.size()}")
                for x in reference, reference_lengths:
                    if x.size(0) != bs:
                        raise ValueError(
                            f"Must have same batch-size: {x.size(0)} != {bs}"
                        )

        # Per utterance mode
        else:
            batch_mode = False

            if speech.dim() > 2:
                raise ValueError(f"Expect (L,) or (L, D): {speech.size()}")
            # (L,) -> (1, L) or (L, D) -> (1, L, D)
            speech = speech[None]

            if reference is not None:
                if reference.dim() > 2:
                    raise ValueError(f"Expect (L,) or (L, D): {reference.size()}")
                # (L,) -> (1, L) or (L, D) -> (1, L, D)
                reference = reference[None]

        # 2. [Option] If waw wave input, extract features
        if speech.dim() == 2:
            # speech: (B, L) -> (B, L2, D)
            speech = torch.tensor(
                self.preprocess_fn("<dummy>", {"speech": speech.numpy()})["speech"]
            )
        if reference is not None:
            if reference.dim() == 2:
                # reference: (B, L) -> (B, L2, D)
                reference = torch.tensor(
                    self.preprocess_fn("<dummy>", {"reference": reference.numpy()})[
                        "reference"
                    ]
                )
            if speech.size(2) != reference.size(2):
                raise RuntimeError(
                    f"Dimension mismatch: {speech.size(2)} != {reference.size(2)}"
                )

        # 3. Compute embed vector
        if reference is None:
            speech = speech.to(getattr(torch, self.dtype))
            batch = dict(speech=speech, speech_lengths=speech_lengths)
            batch = to_device(batch, device=self.device)
            embed_vector = self.model.compute_embed_vector(**batch)
            if not batch_mode:
                return embed_vector.squeeze(0)
            return embed_vector
        else:
            speech = speech.to(getattr(torch, self.dtype))
            reference = reference.to(getattr(torch, self.dtype))
            batch = dict(
                speech=speech,
                speech_lengths=speech_lengths,
                reference=reference,
                reference_lengths=reference_lengths,
            )
            batch = to_device(batch, device=self.device)
            distance, embed_sp, embed_ref = self.model.compute_score(**batch)
            if not batch_mode:
                distance, embed_sp, embed_ref = (
                    distance.squeeze(0),
                    embed_sp.squeeze(0),
                    embed_ref.squeeze(0),
                )
            return distance, embed_sp, embed_ref


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    train_config: str,
    model_file: str,
    allow_variable_data_keys: bool,
):
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

    # 2. Build speech2text
    speaker_ebmed = SPeakerEmbedding(
        train_config=train_config, model_file=model_file, device=device, dtype=dtype,
    )

    # 3. Build data-iterator
    loader = SRETask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=SRETask.build_preprocess_fn(speaker_ebmed.train_args, False),
        collate_fn=SRETask.build_collate_fn(speaker_ebmed.train_args, False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    # 4 .Start for-loop
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datadir_writer = None
    npy_writer = None

    scores = []
    labels = []

    for keys, batch in loader:
        assert isinstance(batch, dict), type(batch)
        assert all(isinstance(s, str) for s in keys), keys
        _bs = len(next(iter(batch.values())))
        assert len(keys) == _bs, f"{len(keys)} != {_bs}"

        # A. Computing score mode
        if "reference" in batch:
            label = batch.pop("label", None)
            score, embed_sp, embed_ref = speaker_ebmed(**batch)

            # At the first loop, initialize DatadirWriter
            if datadir_writer is None:
                datadir_writer = DatadirWriter(output_dir)

            # Write score and label to the files
            for key, s in zip(keys, score):
                datadir_writer["score"][key] = str(s.squeeze().item())
            scores.append(score.detach_().cpu().numpy())

            # NOTE(kamo): Label is not mandatory. If label is given, enable to calculate EER
            if label is not None:
                for key, l in zip(keys, label):
                    datadir_writer["label"][key] = str(l.squeeze(0).item())
                labels.append(label.detach_().squeeze(1).cpu().numpy())

        # B. Dumping embedding vector mode
        else:
            # embed_vector: (B, D)
            embed_vector = speaker_ebmed(**batch)

            # At the first loop, initialize NpyScpWriter
            if npy_writer is None:
                npy_writer = NpyScpWriter(output_dir / "data", output_dir / "embed.scp")

            for key, embed_vector in zip(keys, embed_vector):
                npy_writer[key] = embed_vector

    # A. If computing score mode
    if len(labels) != 0:
        # Compute equal error rate
        labels = np.concatenate(labels)
        assert labels.ndim == 1, labels.shape
        scores = np.concatenate(scores)
        assert scores.ndim == 1, scores.shape
        write_eer(
            output_dir=output_dir, label=labels, score=scores, log_level=log_level
        )


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Compute speaker embedding",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use '_' instead of '-' as separator.
    # '-' is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument("--output_dir", type=str, required=True)
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
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument("--key_file", type=str_or_none)
    group.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    group.add_argument("--train_config", type=str, required=True)
    group.add_argument("--model_file", type=str, required=True)
    group.add_argument(
        "--batch_size", type=int, default=64, help="The batch size for inference",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()

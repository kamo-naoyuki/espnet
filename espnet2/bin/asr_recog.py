#!/usr/bin/env python3
import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import configargparse
import numpy as np
import torch
import yaml
from torch.utils.data.dataloader import DataLoader
from typeguard import check_argument_types

from espnet.nets.beam_search import BeamSearch
from espnet.nets.beam_search import Hypothesis
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.utils.cli_utils import get_commandline_args
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.lm import LMTask
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.dataset import ESPnetDataset
from espnet2.utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter
from espnet2.utils.text_converter import build_text_converter
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none


def recog(
    output_dir: str,
    maxlenratio: float,
    minlenratio: float,
    batch_size: int,
    dtype: str,
    beam_size: int,
    ngpu: int,
    seed: int,
    ctc_weight: float,
    lm_weight: float,
    penalty: float,
    nbest: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: str,
    asr_model_file: str,
    lm_train_config: Optional[str],
    lm_file: Optional[str],
    word_lm_train_config: Optional[str],
    word_lm_file: Optional[str],
    blank_symbol: str,
    token_type: Optional[str],
    bpemodel: Optional[str],
    allow_variable_data_keys: bool,
):
    assert check_argument_types()
    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    if word_lm_train_config is not None:
        raise NotImplementedError("Word LM is not implemented")
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

    # 2. Build ASR model
    scorers = {}
    with Path(asr_train_config).open("r") as f:
        asr_train_args = yaml.load(f, Loader=yaml.Loader)
    asr_train_args = argparse.Namespace(**asr_train_args)
    asr_model = ASRTask.build_model(asr_train_args)
    asr_model.load_state_dict(torch.load(asr_model_file, map_location=device))

    decoder = asr_model.decoder
    ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
    token_list = asr_model.token_list
    scorers.update(
        decoder=decoder, ctc=ctc, length_bonus=LengthBonus(len(token_list)),
    )

    # 3. Build Language model
    if lm_train_config is not None:
        with Path(lm_train_config).open("r") as f:
            lm_train_args = yaml.load(f, Loader=yaml.Loader)
        lm_train_args = argparse.Namespace(**lm_train_args)
        lm = LMTask.build_model(lm_train_args)
        lm.load_state_dict(torch.load(lm_file, map_location=device))
        scorers["lm"] = lm.lm

    # 4. Build BeamSearch object
    weights = dict(
        decoder=1.0 - ctc_weight,
        ctc=ctc_weight,
        lm=lm_weight,
        length_bonus=penalty,
    )
    beam_search = BeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
    )
    beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()
    for scorer in scorers.values():
        if isinstance(scorer, torch.nn.Module):
            scorer.to(device=device, dtype=getattr(torch, dtype)).eval()
    logging.info(f"Beam_search: {beam_search}")
    logging.info(f"Decoding device={device}, dtype={dtype}")

    # 5. Build data-iterator
    dataset = ESPnetDataset(
        data_path_and_name_and_type,
        float_dtype=dtype,
        preprocess=ASRTask.build_preprocess_fn(asr_train_args, False),
    )
    ASRTask.check_task_requirements(dataset, allow_variable_data_keys, False)
    if key_file is None:
        key_file, _, _ = data_path_and_name_and_type[0]

    batch_sampler = ConstantBatchSampler(
        batch_size=batch_size, key_file=key_file, shuffle=False
    )
    logging.info(f"Batch sampler: {batch_sampler}")
    logging.info(f"dataset:\n{dataset}")
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=ASRTask.build_collate_fn(asr_train_args),
        num_workers=num_workers,
    )

    # 6. [Optional] Build Text converter: e.g. bpe-sym -> Text
    if token_type is None:
        token_type = asr_train_args.token_type
    if bpemodel is None:
        bpemodel = asr_train_args.bpemodel
    if token_type is None:
        converter = None
    elif token_type == "bpe":
        if bpemodel is not None:
            converter = build_text_converter(
                token_type=token_type,
                token_list=token_list,
                bpemodel=bpemodel
            )
        else:
            converter = None
    else:
        converter = build_text_converter(
            token_type=token_type,
            token_list=token_list,
        )
    logging.info(f"Text converter: {converter}")

    # 7 .Start for-loop
    # FIXME(kamo): The output format should be discussed about
    with DatadirWriter(output_dir) as writer:
        for keys, batch in loader:
            assert isinstance(batch, dict), type(batch)
            assert all(isinstance(s, str) for s in keys), keys
            _bs = len(next(iter(batch.values())))
            assert len(keys) == _bs, f"{len(keys)} != {_bs}"

            with torch.no_grad():
                # a. To device
                batch = to_device(batch, device)

                # b. Forward Encoder
                enc, _ = asr_model.encode(**batch)
                assert len(enc) == batch_size, len(enc)

                # c. Passed the encoder result and the beam search
                nbest_hyps = beam_search(
                    x=enc[0], maxlenratio=maxlenratio, minlenratio=minlenratio
                )
                nbest_hyps = nbest_hyps[:nbest]

            # Only supporting batch_size==1
            key = keys[0]
            for n in range(1, nbest + 1):
                hyp = nbest_hyps[n - 1]
                assert isinstance(hyp, Hypothesis), type(hyp)

                # remove sos/eos and get results
                token_int = hyp.yseq[1:-1].tolist()
                # Change integer-ids to tokens
                token = [token_list[idx] for idx in token_int]

                # Create a directory: outdir/{n}best_recog
                ibest_writer = writer[f"{n}best_recog"]

                # Write the result to each files
                ibest_writer["token"][key] = " ".join(token).replace(
                    blank_symbol, ""
                )
                ibest_writer["token_int"][key] = " ".join(map(str, token_int))
                ibest_writer["score"][key] = str(hyp.score)

                if converter is not None:
                    text = converter.tokens2text(token)
                    ibest_writer["text"][key] = text


def get_parser():
    parser = configargparse.ArgumentParser(
        description="ASR Decoding",
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

    parser.add_argument("--output_dir", type=str, required=True)
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
    group.add_argument("--asr_train_config", type=str, required=True)
    group.add_argument("--asr_model_file", type=str, required=True)
    group.add_argument("--lm_train_config", type=str)
    group.add_argument("--lm_file", type=str)
    group.add_argument("--word_lm_train_config", type=str)
    group.add_argument("--word_lm_file", type=str)

    group = parser.add_argument_group("Beam-search related")
    group.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )
    group.add_argument(
        "--nbest", type=int, default=1, help="Output N-best hypotheses"
    )
    group.add_argument("--beam_size", type=int, default=20, help="Beam size")
    group.add_argument(
        "--penalty", type=float, default=0.0, help="Insertion penalty"
    )
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain max output length. "
        "If maxlenratio=0.0 (default), it uses a end-detect "
        "function "
        "to automatically find maximum hypothesis lengths",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Input length ratio to obtain min output length",
    )
    group.add_argument(
        "--ctc_weight",
        type=float,
        default=0.5,
        help="CTC weight in joint decoding",
    )
    group.add_argument(
        "--lm_weight", type=float, default=1.0, help="RNNLM weight"
    )
    group.add_argument(
        "--blank_symbol",
        type=str,
        default="<blank>",
        help="The token symbol represents CTC-blank",
    )

    group = parser.add_argument_group("Text converter related")
    group.add_argument(
        "--token_type",
        type=str_or_none,
        default=None,
        choices=["char", "bpe", None],
        help="The token type for ASR model. "
        "If not given, refers from the training args",
    )
    group.add_argument(
        "--bpemodel",
        type=str_or_none,
        default=None,
        help="The model path of sentencepiece. "
        "If not given, refers from the training args",
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    recog(**kwargs)


if __name__ == "__main__":
    main()

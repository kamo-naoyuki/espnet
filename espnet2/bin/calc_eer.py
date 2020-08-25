#!/usr/bin/env python3
import argparse
import logging
import sys
from pathlib import Path
from typing import Union

import matplotlib
import numpy as np
from typeguard import check_argument_types

from espnet.utils.cli_utils import get_commandline_args
from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.sre.utils import calculate_eer
from espnet2.utils import config_argparse


def write_eer(
    output_dir: Union[str, Path],
    log_level: Union[int, str],
    label: Union[str, np.ndarray],
    score: Union[str, np.ndarray],
):
    assert check_argument_types()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(label, str):
        utt2label = load_num_sequence_text(label)
        labels = list(utt2label.values())
    else:
        labels = label
    if isinstance(score, str):
        utt2score = load_num_sequence_text(score, loader_type="csv_float")

        if isinstance(label, str):
            if set(utt2label) != set(utt2score):
                raise RuntimeError(f"Keys mismatches: {label}, {score}")
        scores = list(utt2score.values())
    else:
        scores = score

    if len(scores) <= 1:
        logging.warning("Found less than 1 scores")
        eer, threshold, auc = np.nan, np.nan, np.nan

    elif any(not np.isfinite(s) for s in scores):
        logging.warning("Score contains NaN/Inf values")
        eer, threshold, auc = np.nan, np.nan, np.nan

    else:
        eer, threshold, fpr, tpr, auc = calculate_eer(labels, scores)

        logging.info(f"EER={eer}, Threshold={threshold}, AUC={auc}")

        # Lazy load to avoid the backend error
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.plot(fpr, tpr, label="ROC curve (area = %.2f)" % auc)
        plt.legend()
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.grid(True)
        plt.savefig(output_dir / "roc.png")
        logging.info(f"Generate: {output_dir / 'roc.png'}")

    # Write EER
    with (output_dir / "eer").open("w") as f:
        f.write(f"{eer}\n")
    with (output_dir / "threshold").open("w") as f:
        f.write(f"{threshold}\n")
    with (output_dir / "auc").open("w") as f:
        f.write(f"{auc}\n")


def get_parser():
    parser = config_argparse.ArgumentParser(
        description="Compute equal error rate",
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
    parser.add_argument(
        "--label", type=str, help="A label file having 0 or 1 values", required=True
    )
    parser.add_argument("--score", type=str, help="A score file", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    write_eer(**kwargs)


if __name__ == "__main__":
    main()

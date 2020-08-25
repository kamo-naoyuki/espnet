#!/usr/bin/env python3
from espnet2.tasks.sre import SRETask


def get_parser():
    parser = SRETask.get_parser()
    return parser


def main(cmd=None):
    """LM training.

    Example:

        % python sre_train.py asr --print_config --optim adadelta
        % python sre_train.py --config conf/train_sre.yaml
    """
    SRETask.main(cmd=cmd)


if __name__ == "__main__":
    main()

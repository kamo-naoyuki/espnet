#!/usr/bin/env python3
from espnet2.tasks.dcgan import DCGANTask


def get_parser():
    parser = DCGANTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python dcgan_train.py --print_config --optim adadelta \
                > conf/train_dcgan.yaml
        % python dcgan_train.py --config conf/train_dcgan.yaml
    """
    DCGANTask.main(cmd=cmd)


if __name__ == "__main__":
    main()

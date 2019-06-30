from argparse import Namespace
from typing import Tuple

from chainer.optimizers import Adam as Adam_chainer

from espnet.opts.chainer_backend.opt_interface import OptInterface


def float_pair(string: str) -> Tuple[float, float]:
    pair = string.split(',')
    if len(pair) != 2:
        raise TypeError
    return tuple(float(p) for p in pair)


class Adam(OptInterface):
    @staticmethod
    def add_arguments(parser):
        group = parser.add_argument_group('Optimizer config')
        group.add_argument('--lr', type=float, default=0.001)
        group.add_argument('--adam-weight-decay-rate', type=float, default=0.)
        group.add_argument('--adam-betas', type=float_pair,
                           default=(0.9, 0.999))
        group.add_argument('--adam-eps', type=float, default=1e-3)
        group.add_argument('--adam-eta', type=float, default=1.)
        group.add_argument('--adam-amsgrad', type=bool, default=False)
        group.add_argument('--adam-adabound', type=bool, default=False)
        group.add_argument('--adam-final-lr', type=float, default=0.1)
        group.add_argument('--adam-gamma', type=float, default=0.001)
        return parser

    @staticmethod
    def get(args: Namespace) -> Adam_chainer:
        return Adam_chainer(alpha=args.lr,
                            weight_decay_rate=args.adam_weight_decay_rate,
                            beta1=args.adam_betas[0],
                            beta2=args.adam_betas[1],
                            eps=args.adam_eps,
                            eta=args.adam_eta,
                            amsgrad=args.adam_amsgrad,
                            adabound=args.adam_adabound,
                            final_lr=args.adam_final_lr,
                            gamma=args.adam_gamma,
                            )


# Use espnet.nets.chainer_backend.e2e_asr_transformer.VaswaniRule
# >>> trainer.extend(VaswaniRule(...))
class Noam(OptInterface):
    @staticmethod
    def add_arguments(parser):
        Adam.add_arguments(parser)
        group = parser.add_argument_group('NoamOptimizer config')
        group.add_argument('--noam-warmup', default=25000, type=int,
                           help='noam warmup steps')
        return parser

import argparse
import logging
import random
import shutil
import sys
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import configargparse
import numpy as np
import torch
import torch.nn
import torch.optim
from torch.nn.parallel import data_parallel
from torch.utils.data import DataLoader
from typeguard import check_argument_types
from typeguard import check_return_type
from typing import List

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.utils.cli_utils import get_commandline_args
from espnet2.optimizers.sgd import SGD
from espnet2.schedulers.abs_scheduler import AbsBatchScheduler
from espnet2.schedulers.abs_scheduler import AbsEpochScheduler
from espnet2.schedulers.abs_scheduler import AbsValEpochScheduler
from espnet2.schedulers.noam_lr import NoamLR
from espnet2.train.abs_e2e import AbsE2E
from espnet2.train.add_gradient_noise import add_gradient_noise
from espnet2.train.batch_sampler import AbsSampler
from espnet2.train.batch_sampler import ConstantBatchSampler
from espnet2.train.batch_sampler import SubsetSampler
from espnet2.train.batch_sampler import build_batch_sampler
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.reporter import Reporter
from espnet2.train.reporter import SubReporter
from espnet2.utils.calculate_all_attentions import calculate_all_attentions
from espnet2.utils.device_funcs import to_device
from espnet2.utils.fileio import DatadirWriter
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str2triple_str
from espnet2.utils.types import str_or_none
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


class AbsTask(ABC):
    # Use @staticmethod, or @classmethod,
    # instead of instance method to avoid God classes

    def __init__(self):
        raise RuntimeError("This class can't be instantiated.")

    @classmethod
    @abstractmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Sequence[Dict[str, np.ndarray]]], Dict[str, torch.Tensor]]:
        """Return "collate_fn", which is a callable object and
        will be given to pytorch DataLoader.

        >>> from torch.utils.data import DataLoader
        >>> loader = DataLoader(collate_fn=cls.build_collate_fn(args), ...)

        In many cases, you can use our common collate_fn:

        >>> from espnet2.train.collate_fn import common_collate_fn

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        """Define the required names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_e2e import AbsE2E
        >>> class Model(AbsE2E):
        ...     def forward(self, input, output, opt=None):  pass

        then "required_data_names" should be as

        >>> required_data_names = ('input', 'output')
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        """Define the optional names by Task

        This function is used by
        >>> cls.check_task_requirements()
        If your model is defined as following,

        >>> from espnet2.train.abs_e2e import AbsE2E
        >>> class Model(AbsE2E):
        ...     def forward(self, input, output, opt=None):  pass

        then "optional_data_names" should be as

        >>> optional_data_names = ('opt',)
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_model(cls, args: argparse.Namespace) -> AbsE2E:
        raise NotImplementedError

    @classmethod
    def add_arguments(
        cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="base parser",
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            )

        # NOTE(kamo): Use '_' instead of '-' to avoid confusion.
        #  I think '-' looks really confusing if it's written in yaml.

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        #  to provide --print_config mode. Instead of it, do as
        parser.set_defaults(required=["output_dir"])

        group = parser.add_argument_group("Common configuration")

        group.add_argument(
            "--config", is_config_file=True, help="config file path"
        )
        group.add_argument(
            "--print_config",
            action="store_true",
            help="Print the config file and exit",
        )

        group.add_argument(
            "--log_level",
            type=lambda x: x.upper(),
            default="INFO",
            choices=("INFO", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
            help="The verbose level of logging",
        )

        group.add_argument("--output_dir", type=str_or_none, default=None)
        group.add_argument(
            "--ngpu",
            type=int,
            default=0,
            help="The number of gpus. 0 indicates CPU mode",
        )
        group.add_argument("--seed", type=int, default=0, help="Random seed")
        group.add_argument("--stats_run", type=str2bool, default=False,
                           help="Perform on statistics mode")

        group = parser.add_argument_group("Trainer related")
        group.add_argument(
            "--max_epoch",
            type=int,
            default=40,
            help="The maximum number epoch to train",
        )
        group.add_argument(
            "--train_dtype",
            default="float32",
            choices=["float16", "float32", "float64", "O0", "O1", "O2", "O3"],
            help="Data type for training. "
            "O0,O1,.. flags require apex. "
            "See https://nvidia.github.io/apex/amp.html#opt-levels",
        )
        group.add_argument(
            "--patience",
            type=int_or_none,
            default=None,
            help="Number of epochs to wait without improvement "
            "before stopping the training",
        )
        group.add_argument(
            "--val_scheduler_criterion",
            type=str,
            nargs=2,
            default=("eval", "loss"),
            help="The criterion used for the value given to "
            "the scheduler. Give a pair referring "
            'the phase, "train" or "eval",'
            "and the criterion name. "
            'The mode specifying "min" or "max" can '
            "be changed by --escheduler_conf",
        )
        group.add_argument(
            "--early_stopping_criterion",
            type=str,
            nargs=3,
            default=("eval", "loss", "min"),
            help="The criterion used for judging of "
            "early stopping. Give a pair referring "
            'the phase, "train" or "eval",'
            "the criterion name and the mode, "
            '"min" or "max", e.g. "acc,max".',
        )
        group.add_argument(
            "--best_model_criterion",
            type=str2triple_str,
            action="append",
            default=[
                ("train", "loss", "min"),
                ("eval", "loss", "min"),
                ("train", "acc", "max"),
                ("eval", "acc", "max"),
            ],
            help="The criterion used for judging of "
            "the best model. Give a pair referring "
            'the phase, "train" or "eval",'
            "the criterion name, and "
            'the mode, "min" or "max", e.g. "acc,max".',
        )

        group.add_argument(
            "--grad_clip",
            type=float,
            default=5.0,
            help="Gradient norm threshold to clip",
        )
        group.add_argument(
            "--grad_noise",
            type=str2bool,
            default=False,
            help="The flag to switch to use noise injection to "
            "gradients during training",
        )
        group.add_argument(
            "--accum_grad",
            type=int,
            default=1,
            help="The number of gradient accumulation",
        )
        group.add_argument(
            "--log_interval",
            type=int_or_none,
            default=None,
            help="Show the logs every the number iterations in"
            "each epochs at the training phase. "
            "If None is given, "
            "it is decided according the number "
            "of training samples automatically .",
        )
        group.add_argument(
            "--keep_n_best_snapshot",
            type=int,
            default=10,
            help="Remove previous snapshots excluding "
            "the n-best scored epochs",
        )
        group.add_argument(
            "--num_att_plot",
            type=int,
            default=3,
            help="The number images to plot the outputs "
            "from attention. "
            "This option makes sense "
            "only when attention-based model",
        )
        group.add_argument(
            "--num_workers",
            type=int,
            default=1,
            help="The number of workers used for DataLoader",
        )
        group.add_argument(
            "--no_forward_run",
            type=str2bool,
            default=False,
            help="Just only iterating data loading without "
            "model forwarding and training",
        )
        group.add_argument(
            "--no_backward_run",
            type=str2bool,
            default=False,
            help="Performs data loading and "
            "model forwarding "
            "without backward operations, "
            "optimizer updating, etc.",
        )

        group = parser.add_argument_group(
            "Resuming or transfer learning related"
        )

        def epoch_type(value: str) -> Optional[Union[str, int]]:
            if value == "latest":
                return value
            elif value.lower() in ("none", "null", "nil"):
                return None
            else:
                v = int(value)
                if v < 0:
                    raise TypeError("must be 0 or more integer")
                return v

        egroup = group.add_mutually_exclusive_group()
        egroup.add_argument(
            "--resume_epoch",
            type=epoch_type,
            default=None,
            help="The training starts from the specified epoch. "
            '"latest" indicates the latest-epoch file found '
            "in output_path. If None or 0 are specified, "
            "then training starts from the initial state",
        )
        egroup.add_argument("--resume_path", type=str_or_none, default=None)

        group.add_argument("--pretrain_path", type=str_or_none, default=None)
        group.add_argument("--pretrain_key", type=str_or_none, default=None)

        group = parser.add_argument_group("BatchSampler related")
        group.add_argument(
            "--batch_size",
            type=int,
            default=20,
            help="The mini-batch size used for training",
        )
        group.add_argument(
            "--eval_batch_size",
            type=int_or_none,
            default=None,
            help="If not given, the value of --batch_size is used",
        )

        _batch_type_choices = ("const", "seq", "bin", "frame")
        group.add_argument(
            "--batch_type",
            type=str,
            default="seq",
            choices=_batch_type_choices,
        )
        group.add_argument(
            "--eval_batch_type",
            type=str_or_none,
            default=None,
            choices=_batch_type_choices + (None,),
            help="If not given, the value of --batch_type is used",
        )

        group.add_argument(
            "--train_shape_file", type=str, action="append", default=[]
        )
        group.add_argument(
            "--eval_shape_file", type=str, action="append", default=[]
        )

        group.add_argument(
            "--max_length", type=int, action="append", default=[]
        )

        group.add_argument(
            "--sort_in_batch",
            type=str_or_none,
            default="descending",
            choices=["descending", "ascending", None],
            help="Sort the samples in each mini-batches by the sample "
            "lengths. To enable this, "
            '"shape_file" must have the length information.',
        )
        group.add_argument(
            "--sort_batch",
            type=str_or_none,
            default="descending",
            choices=["descending", "ascending"],
            help="Sort mini-batches by the sample lengths",
        )

        group = parser.add_argument_group("Dataset related")
        group.add_argument(
            "--train_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--eval_data_path_and_name_and_type",
            type=str2triple_str,
            action="append",
            default=[],
        )
        group.add_argument(
            "--allow_variable_data_keys",
            type=str2bool,
            default=False,
            help="Allow the arbitrary keys "
            "for mini-batch with ignoring "
            "the task requirements",
        )

        group = parser.add_argument_group("Optimizer related")
        group.add_argument(
            "--optim",
            type=lambda x: x.lower(),
            default="adadelta",
            choices=cls.optimizer_choices(),
            help="The optimizer type",
        )
        group.add_argument(
            "--optim_conf", action=NestedDictAction, default=dict()
        )

        group.add_argument(
            "--escheduler",
            type=lambda x: str_or_none(x.lower()),
            choices=cls.epoch_scheduler_choices(),
            help="The epoch-scheduler type",
        )
        group.add_argument(
            "--escheduler_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for the epoch scheduler",
        )

        group.add_argument(
            "--bscheduler",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            choices=cls.batch_scheduler_choices(),
            help="The batch-scheduler-type",
        )
        group.add_argument(
            "--bscheduler_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for the batch scheduler",
        )

        assert check_return_type(parser)
        return parser

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return "required", "print_config", "config", "ngpu"

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        assert check_argument_types()
        parser = AbsTask.add_arguments()
        args, _ = parser.parse_known_args()
        config = vars(args)
        # Excludes the options not to be shown
        for k in AbsTask.exclude_opts():
            config.pop(k)

        # Get the default arguments from the specified class
        # e.g. --print_config --optim adadelta
        optim_class = cls.get_optimizer_class(args.optim)
        optim_conf = get_default_kwargs(optim_class)
        optim_conf.update(config["optim_conf"])
        config["optim_conf"] = optim_conf

        if args.escheduler is not None:
            escheduler_class = cls.get_epoch_scheduler_class(args.escheduler)
            escheduler_conf = get_default_kwargs(escheduler_class)
            escheduler_conf.update(config["escheduler_conf"])
            config["escheduler_conf"] = escheduler_conf

        if args.bscheduler is not None:
            bscheduler_class = cls.get_batch_scheduler_class(args.bscheduler)
            bscheduler_conf = get_default_kwargs(bscheduler_class)
            bscheduler_conf.update(config["bscheduler_conf"])
            config["bscheduler_conf"] = bscheduler_conf

        assert check_return_type(config)
        return config

    @classmethod
    def check_required_command_args(cls, args: argparse.Namespace):
        assert check_argument_types()
        for k in vars(args):
            if "-" in k:
                raise RuntimeError(
                    f'Use "_" instead of "-": parser.add_arguments("{k}")'
                )

        required = ", ".join(
            f"--{a}" for a in args.required if getattr(args, a) is None
        )

        if len(required) != 0:
            parser = cls.add_arguments()
            parser.print_help(file=sys.stderr)
            p = Path(sys.argv[0]).name
            print(file=sys.stderr)
            print(
                f"{p}: error: the following arguments are required: "
                f"{required}",
                file=sys.stderr,
            )
            sys.exit(2)

    @classmethod
    def optimizer_choices(cls) -> Tuple[str, ...]:
        choices = (
            "adam",
            "sgd",
            "adadelta",
            "adagrad",
            "adamax",
            "asgd",
            "lbfgs",
            "rmsprop",
            "rprop",
        )
        if LooseVersion(torch.__version__) >= LooseVersion("1.2.0"):
            choices += ("adamw",)
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_optimizer_class(cls, name: str) -> Type[torch.optim.Optimizer]:
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "adam":
            retval = torch.optim.Adam
        elif name.lower() == "sgd":
            retval = SGD
        elif name.lower() == "adadelta":
            retval = torch.optim.Adadelta
        elif name.lower() == "adagrad":
            retval = torch.optim.Adagrad
        elif name.lower() == "adamw":
            retval = torch.optim.AdamW
        elif name.lower() == "adamax":
            retval = torch.optim.Adamax
        elif name.lower() == "asgd":
            retval = torch.optim.ASGD
        elif name.lower() == "lbfgs":
            retval = torch.optim.LBFGS
        elif name.lower() == "rmsprop":
            retval = torch.optim.RMSprop
        elif name.lower() == "rprop":
            retval = torch.optim.Rprop
        else:
            raise RuntimeError(
                f"--optim must be one of {cls.optimizer_choices()}: "
                f"--optim {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def epoch_scheduler_choices(cls) -> Tuple[Optional[str], ...]:
        choices = (
            "ReduceLROnPlateau".lower(),
            "lambdalr",
            "steplr",
            "multisteplr",
            "exponentiallr",
            "CosineAnnealingLR".lower(),
            None,
        )
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_epoch_scheduler_class(cls, name: str) -> Type[AbsEpochScheduler]:
        """Schedulers change optim-parameters at the end of each epochs

        FIXME(kamo): EpochScheduler is confusing name.

        EpochScheduler:
            >>> for epoch in range(10):
            >>>     train(...)
            >>>     scheduler.step()

        ValEpochScheduler:
            >>> for epoch in range(10):
            >>>     train(...)
            >>>     val = validate(...)
            >>>     scheduler.step(val)
        """
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "ReduceLROnPlateau".lower():
            retval = torch.optim.lr_scheduler.ReduceLROnPlateau
        elif name.lower() == "lambdalr":
            retval = torch.optim.lr_scheduler.LambdaLR
        elif name.lower() == "steplr":
            retval = torch.optim.lr_scheduler.StepLR
        elif name.lower() == "multisteplr":
            retval = torch.optim.lr_scheduler.MultiStepLR
        elif name.lower() == "exponentiallr":
            retval = torch.optim.lr_scheduler.ExponentialLR
        elif name.lower() == "CosineAnnealingLR".lower():
            retval = torch.optim.lr_scheduler.CosineAnnealingLR
        else:
            raise RuntimeError(
                f"--escheduler must be one of "
                f"{cls.epoch_scheduler_choices()}: --escheduler {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def batch_scheduler_choices(cls) -> Tuple[Optional[str], ...]:
        choices = (None,)
        if LooseVersion(torch.__version__) >= LooseVersion("1.1.0"):
            choices += ("noamlr",)
        if LooseVersion(torch.__version__) >= LooseVersion("1.3.0"):
            choices += (
                "cycliclr",
                "onecyclelr",
                "CosineAnnealingWarmRestarts".lower(),
            )
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_batch_scheduler_class(cls, name: str) -> Type[AbsBatchScheduler]:
        """Schedulers change optim-parameters after every updating

        FIXME(kamo): BatchScheduler is confusing name.

        BatchScheduler:
            >>> for epoch in range(10):
            >>>     for batch in data_loader:
            >>>         train_batch(...)
            >>>         scheduler.step()
        """
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "cycliclr":
            retval = torch.optim.lr_scheduler.CyclicLR
        elif name.lower() == "onecyclelr":
            retval = torch.optim.lr_scheduler.OneCycleLR
        elif name.lower() == "CosineAnnealingWarmRestarts".lower():
            retval = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        elif name.lower() == "noamlr":
            retval = NoamLR
        else:
            raise RuntimeError(
                f"--bscheduler must be one of "
                f"{cls.batch_scheduler_choices()}: --bscheduler {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def print_config(cls, file=sys.stdout) -> None:
        assert check_argument_types()
        # Shows the config: e.g. python train.py asr --print_config
        config = cls.get_default_config()
        file.write(yaml_no_alias_safe_dump(config, indent=4, sort_keys=False))

    @classmethod
    def main(
        cls, args: argparse.Namespace = None, cmd: Sequence[str] = None
    ) -> None:
        assert check_argument_types()
        print(get_commandline_args(), file=sys.stderr)
        if args is None:
            parser = cls.add_arguments()
            args = parser.parse_args(cmd)
        if args.print_config:
            cls.print_config()
            sys.exit(0)
        cls.check_required_command_args(args)

        logging.basicConfig(
            level=args.log_level,
            format="%(asctime)s (%(module)s:%(lineno)d) "
            "%(levelname)s: %(message)s",
        )

        # 1. Set random-seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.random.manual_seed(args.seed)

        # 2. Build train-data-iterator
        if args.train_dtype in ("float32", "O0", "O1", "O2", "O3"):
            dtype = "float32"
        else:
            dtype = args.train_dtype
        train_dataset = ESPnetDataset(
            args.train_data_path_and_name_and_type,
            float_dtype=dtype,
            preprocess=cls.build_preprocess_fn(args, True),
        )
        cls.check_task_requirements(
            train_dataset, args.allow_variable_data_keys
        )
        train_batch_sampler = build_batch_sampler(
            type=args.batch_type,
            shape_files=args.train_shape_file,
            max_lengths=args.max_length,
            batch_size=args.batch_size,
            shuffle=True,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
        )
        train_iter = DataLoader(
            dataset=train_dataset,
            batch_sampler=train_batch_sampler,
            collate_fn=cls.build_collate_fn(args),
            num_workers=args.num_workers,
        )

        # 3. Build eval-data-iterator
        eval_dataset = ESPnetDataset(
            args.eval_data_path_and_name_and_type,
            float_dtype=dtype,
            preprocess=cls.build_preprocess_fn(args, False),
        )
        cls.check_task_requirements(
            eval_dataset, args.allow_variable_data_keys
        )
        if args.eval_batch_type is None:
            args.eval_batch_type = args.batch_type
        if args.eval_batch_size is None:
            args.eval_batch_size = args.batch_size
        eval_batch_sampler = build_batch_sampler(
            type=args.eval_batch_type,
            shape_files=args.eval_shape_file,
            max_lengths=args.max_length,
            batch_size=args.eval_batch_size,
            shuffle=False,
            sort_in_batch=args.sort_in_batch,
            sort_batch=args.sort_batch,
        )
        eval_iter = DataLoader(
            dataset=eval_dataset,
            batch_sampler=eval_batch_sampler,
            collate_fn=cls.build_collate_fn(args),
            num_workers=args.num_workers,
        )

        # 4. Build a iterator used for attention plot
        if args.num_att_plot != 0:
            plot_attention_sampler = SubsetSampler(
                ConstantBatchSampler(
                    key_file=args.eval_shape_file[0],
                    batch_size=1,
                    shuffle=False,
                ),
                args.num_att_plot,
            )
            plot_attention_iter = DataLoader(
                dataset=eval_dataset,
                batch_sampler=plot_attention_sampler,
                collate_fn=cls.build_collate_fn(args),
                num_workers=args.num_workers,
            )
        else:
            plot_attention_iter = None

        # 5. Build model
        model = cls.build_model(args=args)
        if not isinstance(model, AbsE2E):
            raise RuntimeError(
                f"model must inherit {AbsE2E.__name__}, but got {type(model)}"
            )

        if args.train_dtype in ("float16", "float32", "float64"):
            dtype = getattr(torch, args.train_dtype)
        else:
            dtype = torch.float32
        model = model.to(
            dtype=dtype, device="cuda" if args.ngpu > 0 else "cpu"
        )

        logging.info(f"Model:\n{model}")
        logging.info(f"Train Dataset: {train_dataset}")
        logging.info(f"Train BatchSampler: {train_batch_sampler}")
        logging.info(f"Eval Dataset: {eval_dataset}")
        logging.info(f"Eval BatchSampler: {eval_batch_sampler}")

        # [Stats run]
        if args.stats_run:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "config.yaml").open("w") as f:
                logging.info(
                    f'Saving the configuration in {output_dir / "config.yaml"}'
                )
                yaml_no_alias_safe_dump(vars(args), f, indent=4,
                                        sort_keys=False)
            cls.stats_run(
                model=model,
                train_iter=train_iter,
                eval_iter=eval_iter,
                output_dir=args.output_dir,
                ngpu=args.ngpu,
                log_interval=args.log_interval,
            )
            return

        # 6. Build optimizer
        optimizer_class = cls.get_optimizer_class(args.optim)
        optimizer = optimizer_class(model.parameters(), **args.optim_conf)

        # 7. Build epoch_scheduler: invoked at every epochs
        # e.g. torch.optim.lr_scheduler.StepLR
        if args.escheduler is not None:
            epoch_scheduler_class = cls.get_epoch_scheduler_class(
                args.escheduler
            )
            epoch_scheduler = epoch_scheduler_class(
                optimizer, **args.escheduler_conf
            )
        else:
            epoch_scheduler = None

        # 8. Build batch_scheduler: invoked after every updating
        # e.g. torch.optim.lr_scheduler.CyclicLR
        if args.bscheduler is not None:
            batch_scheduler_class = cls.get_batch_scheduler_class(
                args.bscheduler
            )
            batch_scheduler = batch_scheduler_class(
                optimizer, **args.bscheduler_conf
            )
        else:
            batch_scheduler = None

        logging.info(f"Optimizer:\n{optimizer}")
        logging.info(f"Epoch scheduler: {epoch_scheduler}")
        logging.info(f"Batch scheduler: {batch_scheduler}")

        # 9. Dump "args" to config.yaml
        # NOTE(kamo): "args" should be saved after object-buildings are done
        #  because they are allowed to modify "args".
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "config.yaml").open("w") as f:
            logging.info(
                f'Saving the configuration in {output_dir / "config.yaml"}'
            )
            yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

        reporter = Reporter()

        # 10. Loads states from saved files
        cls.load(
            model=model,
            optimizer=optimizer,
            reporter=reporter,
            output_dir=output_dir,
            batch_scheduler=batch_scheduler,
            epoch_scheduler=epoch_scheduler,
            resume_epoch=args.resume_epoch,
            resume_path=args.resume_path,
            pretrain_path=args.pretrain_path,
            pretrain_key=args.pretrain_key,
            map_location="cuda" if args.ngpu > 0 else "cpu",
        )

        # 11. Start training
        cls.run(
            model=model,
            optimizer=optimizer,
            train_iter=train_iter,
            eval_iter=eval_iter,
            plot_attention_iter=plot_attention_iter,
            reporter=reporter,
            output_dir=output_dir,
            batch_scheduler=batch_scheduler,
            epoch_scheduler=epoch_scheduler,
            ngpu=args.ngpu,
            max_epoch=args.max_epoch,
            train_dtype=args.train_dtype,
            patience=args.patience,
            grad_noise=args.grad_noise,
            accum_grad=args.accum_grad,
            grad_clip=args.grad_clip,
            log_interval=args.log_interval,
            keep_n_best_snapshot=args.keep_n_best_snapshot,
            early_stopping_criterion=args.early_stopping_criterion,
            best_model_criterion=args.best_model_criterion,
            val_scheduler_criterion=args.val_scheduler_criterion,
            no_forward_run=args.no_forward_run,
            no_backward_run=args.no_backward_run,
        )

    @classmethod
    def check_task_requirements(
        cls,
        dataset: ESPnetDataset,
        allow_variable_data_keys: bool,
        train: bool = True,
    ) -> None:
        """Check if the dataset satisfy the requirement of current Task"""
        assert check_argument_types()
        mes = (
            f"If you intend to use an additional input, modify "
            f'"{cls.__name__}.required_data_names()" or '
            f'"{cls.__name__}.optional_data_names()". '
            f"Otherwise you need to set --allow_variable_data_keys true "
        )

        for k in cls.required_data_names(train):
            if not dataset.has_name(k):
                raise RuntimeError(
                    f'"{cls.required_data_names(train)}" are required for'
                    f' {cls.__name__}. but "{dataset.names()}" are input.\n'
                    f"{mes}"
                )
        if not allow_variable_data_keys:
            task_keys = cls.required_data_names(
                train
            ) + cls.optional_data_names(train)
            for k in dataset.names():
                if k not in task_keys:
                    raise RuntimeError(
                        f"The data-name must be one of {task_keys} "
                        f'for {cls.__name__}: "{k}" is not allowed.\n{mes}'
                    )

    @classmethod
    def load(
        cls,
        model: AbsE2E,
        optimizer: torch.optim.Optimizer,
        reporter: Reporter,
        output_dir: Union[str, Path],
        batch_scheduler: Optional[AbsBatchScheduler],
        epoch_scheduler: Optional[AbsEpochScheduler],
        resume_epoch: Optional[Union[int, str]],
        resume_path: Optional[Union[str, Path]],
        pretrain_path: Optional[Union[str, Path]],
        pretrain_key: Optional[str],
        map_location: str,
    ) -> None:
        assert check_argument_types()
        # For resuming: Specify either resume_epoch or resume_path.
        #     - resume_epoch: Load from outdir/{}epoch/.
        #     - resume_path: Load from the specified path.
        # Find the latest epoch snapshot
        if resume_epoch == "latest":
            resume_epoch = 0
            latest = None
            for p in output_dir.glob("*epoch/timestamp"):
                try:
                    n = int(p.parent.name.replace("epoch", ""))
                except TypeError:
                    continue
                with p.open("r") as f:
                    # Read the timestamp and comparing
                    date = f.read().strip()
                    try:
                        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                if latest is None or date > latest:
                    resume_epoch = n
                    latest = date

        # If not found any snapshots, then nothing is done
        if resume_epoch == 0:
            resume_epoch = None

        if resume_epoch is not None or resume_path is not None:
            if resume_path is None:
                resume_path = output_dir / f"{resume_epoch}epoch"
                logging.info(
                    f"--resume_epoch {resume_epoch}: "
                    f"Loading from {resume_path}"
                )
            else:
                logging.info(
                    f"--resume_path {resume_path}: "
                    f"Loading from {resume_path}"
                )

            for key, obj in [
                ("model", model),
                ("optimizer", optimizer),
                ("reporter", reporter),
                ("epoch_scheduler", epoch_scheduler),
                ("batch_scheduler", batch_scheduler),
            ]:
                _st = torch.load(
                    resume_path / f"{key}.pt", map_location=map_location
                )
                if obj is not None:
                    obj.load_state_dict(_st)

        # FIXME(kamo): Should be done in Task.build_model() or in model?
        # For distillation, fine-tuning, transfer learning, etc.
        if pretrain_path is not None:
            if pretrain_key is None:
                obj = model
            else:

                def get_attr(obj: Any, key: str):
                    """

                    >>> class A(torch.nn.Module):
                    ...     def __init__(self):
                    ...         super().__init__()
                    ...         self.linear = torch.nn.Linear(10, 10)
                    >>> a = A()
                    >>> assert A.linear.weight is get_attr(A, 'linear.weight')

                    """
                    if key.strip() == "":
                        return obj
                    for k in key.split("."):
                        obj = getattr(obj, k)
                    return obj

                obj = get_attr(model, pretrain_key)

            state_dict = obj.state_dict()
            pretrained_dict = torch.load(
                pretrain_path, map_location=map_location
            )
            # Ignores the parameters not existing in the train-model
            pretrained_dict = {
                k: v for k, v in pretrained_dict.items() if k in state_dict
            }
            state_dict.update(pretrained_dict)
            obj.load_state_dict(state_dict)

    @classmethod
    @torch.no_grad()
    def stats_run(
        cls,
        model: AbsE2E,
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        output_dir: Union[str, Path],
        ngpu: Optional[int],
        log_interval: Optional[int],
    ) -> None:
        """Running for deriving the shape information
        from data and gathering stats
        """
        assert check_argument_types()
        output_dir = Path(output_dir)

        for itr, mode in zip([train_iter, eval_iter], ["train", "eval"]):
            if log_interval is None:
                log_interval = max(len(itr) // 20, 10)

            count = 0
            sum_dict = defaultdict(lambda: 0)
            sq_dict = defaultdict(lambda: 0)

            with DatadirWriter(output_dir / mode) as writer:
                for iiter, (keys, batch) in enumerate(itr, 1):
                    batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
                    count += len(next(iter(batch.values())))

                    # 1. Write shape file
                    for name in batch:
                        if name.endswith("_lengths"):
                            continue
                        for i, (k, data) in enumerate(zip(keys, batch[name])):
                            if f"{name}_lengths" in batch:
                                lg = int(batch[f"{name}_lengths"][i])
                                shape = ",".join(
                                    map(str, (lg,) + data.shape[1:])
                                )
                            else:
                                shape = ",".join(map(str, data.shape))
                            writer[f"{name}_shape"][k] = shape

                    # 2. Extract feats and calc sum and square sum
                    data = model.collect_feats(**batch)
                    for k, v in data.items():
                        if k.endswith("_lengths"):
                            continue
                        if f"{k}_lengths" in data:
                            # value: (Batch, Length, Dim, ...)
                            ind = (0, 1)
                        else:
                            ind = 0
                        v = v.cpu()
                        v.masked_fill_(
                            make_pad_mask(data[f"{k}_lengths"], v, 1), 0.0
                        )
                        sum_dict[k] += v.sum(ind).cpu().numpy()
                        sq_dict[k] += (v ** 2).sum(ind).cpu().numpy()

                    if iiter % log_interval == 0:
                        logging.info(f"Niter: {iiter}")

            for key in sum_dict:
                np.savez(
                    output_dir / mode / f"{key}_stats.npz",
                    count=count,
                    sum=sum_dict[key],
                    sum_square=sq_dict[key]
                )
            with (output_dir / mode / "shape_keys").open("w") as f:
                f.write("\n".join(filter(lambda x: not x.endswith("_lengths"), batch)) + "\n")
            with (output_dir / mode / "stats_keys").open("w") as f:
                f.write("\n".join(sum_dict) + "\n")

    @classmethod
    def run(
        cls,
        model: AbsE2E,
        optimizer: torch.optim.Optimizer,
        train_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        eval_iter: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        plot_attention_iter,
        reporter: Reporter,
        output_dir: Union[str, Path],
        batch_scheduler: Optional[AbsBatchScheduler],
        epoch_scheduler: Optional[AbsEpochScheduler],
        max_epoch: int,
        patience: Optional[int],
        ngpu: int,
        train_dtype: str,
        grad_noise: bool,
        accum_grad: int,
        grad_clip: float,
        log_interval: Optional[int],
        keep_n_best_snapshot: int,
        early_stopping_criterion: Sequence[str],
        best_model_criterion: Sequence[Sequence[str]],
        val_scheduler_criterion: Sequence[str],
        no_forward_run: bool,
        no_backward_run: bool,
    ) -> None:
        assert check_argument_types()

        # For apex supporting
        if train_dtype in ("O0", "O1", "O2", "O3"):
            try:
                from apex import amp
            except ImportError:
                logging.error(
                    f"You need to install apex. "
                    f"See https://github.com/NVIDIA/apex#linux"
                )
                raise
            model, optimizer = amp.initialize(
                model, optimizer, opt_level=train_dtype
            )

        start_epoch = reporter.get_epoch() + 1
        if start_epoch == max_epoch + 1:
            logging.warning(
                f"The training has already reached at "
                f"max_epoch: {start_epoch}"
            )

        best_epoch_dict = {}
        for iepoch in range(start_epoch, max_epoch + 1):
            logging.info(f"{iepoch}epoch started")

            reporter.set_epoch(iepoch)
            # 1. Train and eval for one-epoch
            with reporter.observe("train") as sub_reporter:
                all_steps_are_invalid = cls.train(
                    model=model,
                    optimizer=optimizer,
                    iterator=train_iter,
                    reporter=sub_reporter,
                    scheduler=batch_scheduler,
                    ngpu=ngpu,
                    use_apex=train_dtype in ("O0", "O1", "O2", "O3"),
                    grad_noise=grad_noise,
                    accum_grad=accum_grad,
                    grad_clip=grad_clip,
                    log_interval=log_interval,
                    no_forward_run=no_forward_run,
                    no_backward_run=no_backward_run,
                )
            with reporter.observe("eval") as sub_reporter:
                cls.eval(
                    model=model,
                    iterator=eval_iter,
                    reporter=sub_reporter,
                    ngpu=ngpu,
                    no_forward_run=no_forward_run,
                )
            if plot_attention_iter is not None and not no_forward_run:
                with reporter.observe("att_plot") as sub_reporter:
                    cls.plot_attention(
                        model=model,
                        output_dir=output_dir / "att_ws" / f"{iepoch}epoch",
                        iterator=plot_attention_iter,
                        ngpu=ngpu,
                        reporter=sub_reporter,
                    )

            # 2. Scheduler step
            #   Controls opt-params by scheduler e.g. learning rate decay
            if epoch_scheduler is not None:
                if isinstance(epoch_scheduler, AbsValEpochScheduler):
                    _phase, _criterion = val_scheduler_criterion
                    if not reporter.has(_phase, _criterion):
                        raise RuntimeError(
                            f"{_phase}.{_criterion} is not found in stats"
                            f"{reporter.get_all_keys()}"
                        )
                    val = reporter.get_value(_phase, _criterion)
                    epoch_scheduler.step(val)
                else:
                    epoch_scheduler.step()

            # 3. Report the results
            reporter.logging()
            reporter.save_stats_plot(output_dir / "stats")

            # 4. Save the snapshot
            for key, obj in [
                ("model", model),
                ("optimizer", optimizer),
                ("reporter", reporter),
                ("epoch_scheduler", epoch_scheduler),
                ("batch_scheduler", batch_scheduler),
            ]:
                (output_dir / f"{iepoch}epoch").mkdir(
                    parents=True, exist_ok=True
                )
                p = output_dir / f"{iepoch}epoch" / f"{key}.pt"
                p.parent.mkdir(parents=True, exist_ok=True)
                torch.save(obj.state_dict() if obj is not None else None, p)
            # Write the datetime in "timestamp"
            with (output_dir / f"{iepoch}epoch" / "timestamp").open("w") as f:
                dt = datetime.now()
                f.write(dt.strftime("%Y-%m-%d %H:%M:%S") + "\n")

            # 5. Saves the best model
            _improved = []
            for _phase, k, _mode in best_model_criterion:
                if reporter.has(_phase, k):
                    best_epoch, _ = reporter.sort_epochs_and_values(
                        _phase, k, _mode
                    )[0]
                    best_epoch_dict[(_phase, k)] = best_epoch
                    # Creates sym links if it's the best result
                    if best_epoch == iepoch:
                        p = output_dir / f"{_phase}.{k}.best.pt"
                        if p.is_symlink() or p.exists():
                            p.unlink()
                        p.symlink_to(Path(f"{iepoch}epoch") / f"model.pt")
                        _improved.append(f"{_phase}.{k}")
            if len(_improved) == 0:
                logging.info(f"There are no improvements in this epoch")
            else:
                logging.info(
                    f"The best model has been updated: " + ", ".join(_improved)
                )

            # 6. Remove the snapshot excluding n-best and the current epoch
            _removed = []
            # nbests: List[List[Tuple[epoch, value]]]
            nbests = [
                reporter.sort_epochs_and_values(ph, k, m)[
                    :keep_n_best_snapshot
                ]
                for ph, k, m in best_model_criterion
                if reporter.has(ph, k)
            ]
            # nbests: Set[epoch]
            if len(nbests) != 0:
                nbests = set.union(*[set(i[0] for i in v) for v in nbests])
            else:
                nbests = set()
            for e in range(1, iepoch):
                p = output_dir / f"{e}epoch"
                if p.exists() and e not in nbests:
                    shutil.rmtree(p)
                    _removed.append(str(p))
            if len(_removed) != 0:
                logging.info(
                    f"The snapshot was removed: " + ", ".join(_removed)
                )

            # 7. If any updating haven't happen, stops the training
            if all_steps_are_invalid:
                logging.warning(
                    f"The gradients at all steps are invalid "
                    f"in this epoch. Something seems wrong. "
                    f"This training was stopped at {iepoch}epoch"
                )
                break

            # 8. Check early stopping
            if patience is not None:
                _phase, _criterion, _mode = early_stopping_criterion
                if not reporter.has(_phase, _criterion):
                    raise RuntimeError(
                        f"{_phase}.{_criterion} is not found in stats: "
                        f"{reporter.get_all_keys()}"
                    )
                best_epoch, _ = reporter.sort_epochs_and_values(
                    _phase, _criterion, _mode
                )[0]
                if iepoch - best_epoch > patience:
                    logging.info(
                        f"[Early stopping] {_phase}.{_criterion} "
                        f"has not been improved "
                        f"{iepoch - best_epoch} epochs continuously. "
                        f"The training was stopped at {iepoch}epoch"
                    )
                    break

        else:
            logging.info(f"The training was finished at {max_epoch} epochs ")

        # 9. Average the n-best models
        cls.average_nbest_models(
            reporter=reporter,
            output_dir=output_dir,
            best_model_criterion=best_model_criterion,
            nbest=keep_n_best_snapshot,
        )

    @classmethod
    def train(
        cls,
        model: AbsE2E,
        iterator: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        optimizer: torch.optim.Optimizer,
        reporter: SubReporter,
        scheduler: Optional[AbsBatchScheduler],
        ngpu: int,
        use_apex: bool,
        grad_noise: bool,
        accum_grad: int,
        grad_clip: float,
        log_interval: Optional[int],
        no_forward_run: bool,
        no_backward_run: bool,
    ) -> bool:
        assert check_argument_types()
        if log_interval is None:
            log_interval = max(len(iterator) // 20, 10)

        model.train()
        all_steps_are_invalid = True
        for iiter, (_, batch) in enumerate(iterator, 1):
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                all_steps_are_invalid = False
                reporter.register({})
                continue

            if ngpu <= 1:
                # NOTE(kamo): data_parallel also should work with ngpu=1,
                # but for debuggability it's better to keep this block.
                loss, stats, weight = model(**batch)
            else:
                loss, stats, weight = data_parallel(
                    model, (), range(ngpu), module_kwargs=batch
                )
                # Weighted averaging of loss from torch-data-parallel
                loss = (loss * weight.to(loss.dtype)).sum(0) / weight.sum()
                stats = {
                    k: (v * weight.to(v.dtype)).sum(0) / weight.sum()
                    if v is not None
                    else None
                    for k, v in stats.items()
                }
                weight = weight.sum()
            reporter.register(stats, weight)

            if no_backward_run:
                all_steps_are_invalid = False
                continue

            if use_apex:
                from apex import amp

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            del loss

            # gradient noise injection
            if grad_noise:
                add_gradient_noise(
                    model,
                    reporter.get_total_count(),
                    duration=100,
                    eta=1.0,
                    scale_factor=0.55,
                )

            # compute the gradient norm to check if it is normal or not
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_clip
            )
            if iiter % accum_grad == 0:
                if not np.isfinite(grad_norm):
                    logging.warning(
                        f"The grad norm is {grad_norm}. "
                        f"Skipping updating the model."
                    )
                else:
                    all_steps_are_invalid = False
                    optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                # Register lr
                reporter.register(
                    {
                        f"lr_{i}": pg["lr"]
                        for i, pg in enumerate(optimizer.param_groups)
                        if "lr" in pg
                    },
                    not_increment_count=True,
                )

            if iiter % log_interval == 0:
                reporter.logging(nlatest=log_interval)
        return all_steps_are_invalid

    @classmethod
    @torch.no_grad()
    def eval(
        cls,
        model: AbsE2E,
        iterator: DataLoader and Iterable[Dict[str, torch.Tensor]],
        reporter: SubReporter,
        ngpu: int,
        no_forward_run: bool,
    ) -> None:
        assert check_argument_types()
        model.eval()
        for (_, batch) in iterator:
            assert isinstance(batch, dict), type(batch)
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")
            if no_forward_run:
                reporter.register({})
                continue

            if ngpu <= 1:
                _, stats, weight = model(**batch)
            else:
                _, stats, weight = data_parallel(
                    model, (), range(ngpu), module_kwargs=batch
                )
                stats = {
                    k: (v * weight.to(v.dtype)).sum(0) / weight.sum()
                    if v is not None
                    else None
                    for k, v in stats.items()
                }
                weight = weight.sum()

            reporter.register(stats, weight)

    @classmethod
    @torch.no_grad()
    def plot_attention(
        cls,
        model: AbsE2E,
        output_dir: Path,
        iterator: DataLoader and Iterable[Tuple[List[str], Dict[str, torch.Tensor]]],
        ngpu: int,
        reporter: SubReporter,
    ) -> None:
        assert check_argument_types()
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        model.eval()
        output_dir = Path(output_dir)
        for ids, batch in iterator:
            assert isinstance(batch, dict), type(batch)
            assert len(next(iter(batch.values()))) == len(ids), (
                len(next(iter(batch.values()))),
                len(ids),
            )
            batch = to_device(batch, "cuda" if ngpu > 0 else "cpu")

            # 1. Forwarding model and gathering all attentions
            #    calculate_all_attentions() uses single gpu only.
            att_dict = calculate_all_attentions(model, batch)

            # 2. Plot attentions: This part is slow due to matplotlib
            for k, att_list in att_dict.items():
                assert len(att_list) == len(ids), (len(att_list), len(ids))
                for id_, att_w in zip(ids, att_list):

                    if isinstance(att_w, torch.Tensor):
                        att_w = att_w.detach().cpu().numpy()

                    if att_w.ndim == 2:
                        att_w = att_w[None]
                    elif att_w.ndim > 3 or att_w.ndim == 1:
                        raise RuntimeError(
                            f"Must be 2 or 3 dimension: {att_w.ndim}"
                        )

                    w, h = plt.figaspect(1.0 / len(att_w))
                    fig = plt.Figure(figsize=(w * 1.3, h * 1.3))
                    axes = fig.subplots(1, len(att_w))
                    if len(att_w) == 1:
                        axes = [axes]

                    for ax, aw in zip(axes, att_w):
                        ax.imshow(aw.astype(np.float32), aspect="auto")
                        ax.set_title(f"{k}_{id_}")
                        ax.set_xlabel("Input")
                        ax.set_ylabel("Output")
                        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

                    p = output_dir / id_ / (k + ".png")
                    p.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(p)

                    # Dummy register() stimulates to increment the counter
                    reporter.register({})

    @classmethod
    @torch.no_grad()
    def average_nbest_models(
        cls,
        output_dir: Path,
        reporter: Reporter,
        best_model_criterion: Sequence[Sequence[str]],
        nbest: int,
    ) -> None:
        assert check_argument_types()
        # 1. Get nbests: List[Tuple[str, str, List[Tuple[epoch, value]]]]
        nbest_epochs = [
            (ph, k, reporter.sort_epochs_and_values(ph, k, m)[:nbest])
            for ph, k, m in best_model_criterion
            if reporter.has(ph, k)
        ]

        _loaded = {}
        for ph, cr, epoch_and_values in nbest_epochs:
            # Note that len(epoch_and_values) doesn't always equal to nbest.

            op = output_dir / f"{ph}.{cr}.ave_{len(epoch_and_values)}best.pt"
            logging.info(
                f"Averaging {len(epoch_and_values)}best models: "
                f'criterion="{ph}.{cr}": {op}'
            )

            avg = None
            # 2.a Averaging model
            for e, _ in epoch_and_values:
                if e not in _loaded:
                    _loaded[e] = torch.load(
                        output_dir / f"{e}epoch" / "model.pt",
                        map_location="cpu",
                    )
                states = _loaded[e]

                if avg is None:
                    avg = states
                else:
                    # Accumulated
                    for k in avg:
                        avg[k] += states[k]
            for k in avg:
                avg[k] /= len(epoch_and_values)

            # 2.b Save the ave model and create a symlink
            torch.save(avg, op)
            sym_op = output_dir / f"{ph}.{cr}.ave.pt"
            if sym_op.is_symlink() or sym_op.exists():
                sym_op.unlink()
            sym_op.symlink_to(op.name)

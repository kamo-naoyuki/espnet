import argparse
import logging
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

import configargparse
import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.lm.abs_model import AbsLM
from espnet2.lm.e2e import LanguageE2E
from espnet2.lm.seq_rnn import SequentialRNNLM
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.initialize import initialize
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


class LMTask(AbsTask):
    @classmethod
    def add_arguments(
        cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="Train language model",
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            )

        AbsTask.add_arguments(parser)
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["token_list"]

        group.add_argument(
            "--token_list",
            type=str_or_none,
            default=None,
            help="A text mapping int-id to token",
        )
        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=cls.init_choices(),
        )
        group.add_argument(
            "--lm",
            type=lambda x: x.lower(),
            default="seq_rnn",
            choices=cls.lm_choices(),
            help="Specify lm class",
        )
        group.add_argument(
            "--lm_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for lm class.",
        )
        group.add_argument(
            "--e2e_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for E2E class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--use_preprocessor",
            type=str2bool,
            default=False,
            help="Apply preprocessing to data or not",
        )
        group.add_argument(
            "--token_type",
            type=str,
            default="bpe",
            choices=["bpe", "char", "word"],
            help="",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file fo sentencepiece",
        )

        assert check_return_type(parser)
        return parser

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        return AbsTask.exclude_opts()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        assert check_argument_types()
        # This method is used only for --print_config

        # 0. Parse command line arguments
        parser = LMTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        lm_class = cls.get_lm_class(args.lm)
        lm_conf = get_default_kwargs(lm_class)
        e2e_conf = get_default_kwargs(LanguageE2E)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        lm_conf.update(config["lm_conf"])
        e2e_conf.update(config["e2e_conf"])

        # 5. Reassign them to the configuration
        config.update(lm_conf=lm_conf, e2e_conf=e2e_conf)

        # 6. Excludes the options not to be shown
        for k in cls.exclude_opts():
            config.pop(k)

        assert check_return_type(config)
        return config

    @classmethod
    def init_choices(cls) -> Tuple[Optional[str], ...]:
        choices = (
            "chainer",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            None,
        )
        return choices

    @classmethod
    def lm_choices(cls) -> Tuple[str, ...]:
        choices = ("seq_rnn",)
        return choices

    @classmethod
    def get_lm_class(cls, name: str) -> Type[AbsLM]:
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "seq_rnn":
            retval = SequentialRNNLM
        else:
            raise RuntimeError(
                f"--lm must be one of " f"{cls.lm_choices()}: --lm {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                  Tuple[List[str], Dict[str, torch.Tensor]]]:
        assert check_argument_types()
        return CommonCollateFn(int_pad_value=0)

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        if args.use_preprocessor:
            retval = CommonPreprocessor(
                train=train,
                token_type=args.token_type,
                token_list=args.token_list,
                bpemodel=args.bpemodel,
            )
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        retval = ()
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> LanguageE2E:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]

            # "args" is saved as it is in a yaml file by BaseTask.main().
            # Overwriting token_list to keep it as "portable".
            args.token_list = token_list.copy()
        elif isinstance(args.token_list, (tuple, list)):
            token_list = args.token_list.copy()
        else:
            raise RuntimeError("token_list must be str or dict")

        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. Build LM model
        lm_class = cls.get_lm_class(args.lm)
        lm = lm_class(vocab_size=vocab_size, **args.lm_conf)

        # 2. Build E2E
        # Assume the last-id is sos_and_eos
        model = LanguageE2E(lm=lm, vocab_size=vocab_size, **args.e2e_conf)

        # FIXME(kamo): Should be done in model?
        # 3. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

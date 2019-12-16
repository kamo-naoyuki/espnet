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

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.inversible_interface import InversibleInterface
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.tts.abs_model import AbsTTS
from espnet2.tts.e2e import TTSE2E
from espnet2.tts.tacotron2 import Tacotron2
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str_or_none


class TTSTask(AbsTask):
    @classmethod
    def add_arguments(
        cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="Train launguage model",
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
        excl = group.add_mutually_exclusive_group()
        excl.add_argument(
            "--odim",
            type=int_or_none,
            default=None,
            help="The number of dimension of output feature",
        )
        excl.add_argument(
            "--feats_extract",
            type=lambda x: str_or_none(x.lower()),
            default="default",
            choices=cls.feats_extract_choices(),
            help="Specify feats_extract class",
        )
        group.add_argument(
            "--feats_extract_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for feats_extract class.",
        )
        group.add_argument(
            "--normalize",
            type=lambda x: str_or_none(x.lower()),
            default="global_mvn",
            choices=cls.normalize_choices(),
            help="Specify normalization class",
        )
        group.add_argument(
            "--normalize_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for normalization class.",
        )
        group.add_argument(
            "--tts",
            type=lambda x: x.lower(),
            default="tacotron2",
            choices=cls.tts_choices(),
            help="Specify tts class",
        )
        group.add_argument(
            "--tts_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for TTS class.",
        )
        group.add_argument(
            "--e2e_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for E2E class.",
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
        parser = TTSTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        if args.odim is None:
            feats_extract_class = cls.get_feats_extract_class(
                args.feats_extract
            )
            feats_extract_conf = get_default_kwargs(feats_extract_class)
        else:
            if hasattr(args, "feats_extract"):
                # Either one of feats_extract and odim can be selected
                delattr(args, "feats_extract")
            feats_extract_conf = {}
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize_conf = get_default_kwargs(normalize_class)
        else:
            normalize_conf = None

        tts_class = cls.get_tts_class(args.tts)
        tts_conf = get_default_kwargs(tts_class)
        e2e_conf = get_default_kwargs(TTSE2E)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        feats_extract_conf.update(config["feats_extract_conf"])
        normalize_conf.update(config["normalize_conf"])
        tts_conf.update(config["tts_conf"])
        e2e_conf.update(config["e2e_conf"])

        # 5. Reassign them to the configuration
        config.update(
            feats_extract_conf=feats_extract_conf,
            normalize_conf=normalize_conf,
            tts_conf=tts_conf,
            e2e_conf=e2e_conf,
        )

        # 6. Excludes the options not to be shown
        for k in cls.exclude_opts():
            config.pop(k)

        assert check_return_type(config)
        return config

    @classmethod
    def feats_extract_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ("default", None)
        return choices

    @classmethod
    def get_feats_extract_class(cls, name: str) -> Type[AbsFrontend]:
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "default":
            # FIXME(kamo): Implement TTS feature extractor, not using ASR?
            # Fetch the frontend module from Asr as feature extractor
            retval = DefaultFrontend
        else:
            raise RuntimeError(
                f"--feats_extract must be one of "
                f"{cls.feats_extract_choices()}: --feats_extract {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def normalize_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ("global_mvn", None)
        return choices

    @classmethod
    def get_normalize_class(
        cls, name: str
    ) -> Type[AbsNormalize and InversibleInterface]:
        assert check_argument_types()
        if name.lower() == "global_mvn":
            retval = GlobalMVN
        else:
            raise RuntimeError(
                f"--normalize must be one of "
                f"{cls.normalize_choices()}: --normalize {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def tts_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ("tacotron2",)
        return choices

    @classmethod
    def get_tts_class(cls, name: str) -> Type[AbsTTS]:
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "tacotron2":
            retval = Tacotron2
        else:
            raise RuntimeError(
                f"--tts must be one of " f"{cls.tts_choices()}: --tts {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                  Tuple[List[str], Dict[str, torch.Tensor]]]:
        assert check_argument_types()
        return CommonCollateFn(
            float_pad_value=0.0, int_pad_value=0, not_sequence=["spembs"]
        )

    @classmethod
    def build_preprocess_fn(
        cls, args: argparse.Namespace, train: bool
    ) -> Optional[Callable[[str, Dict[str, np.array]], Dict[str, np.ndarray]]]:
        assert check_argument_types()
        return None

    @classmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        if train:
            retval = ("text", "speech")
        else:
            # Inference mode
            retval = ("text",)
        return retval

    @classmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        if train:
            retval = ("spembs", "spcs")
        else:
            # Inference mode
            retval = ("spembs",)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> TTSE2E:
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

        # 1. feats_extract
        if args.odim is None:
            feats_extract_class = cls.get_feats_extract_class(
                args.feats_extract
            )
            feats_extract = feats_extract_class(**args.feats_extract_conf)
            odim = feats_extract.out_dim()
        else:
            if hasattr(args, "feats_extract"):
                # Either one of feats_extract and odim can be selected
                delattr(args, "feats_extract")
            args.feats_extract_conf = {}
            feats_extract = None
            odim = args.odim

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. TTS
        tts_class = cls.get_tts_class(args.tts)
        tts = tts_class(idim=vocab_size, odim=odim, **args.tts_conf)

        # 4. Build model
        model = TTSE2E(
            feats_extract=feats_extract,
            normalize=normalize,
            tts=tts,
            **args.e2e_conf,
        )
        assert check_return_type(model)
        return model

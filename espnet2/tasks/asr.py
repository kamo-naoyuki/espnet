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

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.decoder.rnn_decoder import RNNDecoder
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.e2e import ASRE2E
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.rnn_encoder import RNNEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.encoder.vgg_rnn_encoder import VGGRNNEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.initialize import initialize
from espnet2.train.preprocessor import CommonPreprocessor
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import int_or_none
from espnet2.utils.types import str2bool
from espnet2.utils.types import str_or_none


class ASRTask(AbsTask):
    @classmethod
    def add_arguments(
        cls, parser: configargparse.ArgumentParser = None
    ) -> configargparse.ArgumentParser:
        assert check_argument_types()
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description="Train ASR",
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

        excl = group.add_mutually_exclusive_group()
        excl.add_argument(
            "--input_size",
            type=int_or_none,
            default=None,
            help="The number of input dimension of the feature",
        )
        excl.add_argument(
            "--frontend",
            type=lambda x: str_or_none(x.lower()),
            default="default",
            choices=cls.frontend_choices(),
            help="Specify frontend class",
        )
        group.add_argument(
            "--frontend_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for frontend class.",
        )
        group.add_argument(
            "--normalize",
            type=lambda x: str_or_none(x.lower()),
            default="utterance_mvn",
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
            "--encoder",
            type=lambda x: x.lower(),
            default="vgg_rnn",
            choices=cls.encoder_choices(),
            help="Specify Encoder type",
        )
        group.add_argument(
            "--encoder_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for Encoder class.",
        )

        group.add_argument(
            "--decoder",
            type=lambda x: x.lower(),
            default="rnn",
            choices=cls.decoder_choices(),
            help="Specify Decoder type",
        )
        group.add_argument(
            "--decoder_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for Decoder class.",
        )

        group.add_argument(
            "--ctc_conf",
            action=NestedDictAction,
            default=dict(),
            help="The keyword arguments for CTC class.",
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
            help="The text will be tokenized " "in the specified level token",
        )
        group.add_argument(
            "--bpemodel",
            type=str_or_none,
            default=None,
            help="The model file of sentencepiece",
        )
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
        parser = ASRTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        if args.input_size is None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend_conf = get_default_kwargs(frontend_class)
        else:
            if hasattr(args, "frontend"):
                # Either one of frontend and input_size can be selected
                delattr(args, "frontend")
            frontend_conf = {}
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize_conf = get_default_kwargs(normalize_class)
        else:
            normalize_conf = None

        encoder_class = cls.get_encoder_class(args.encoder)
        encoder_conf = get_default_kwargs(encoder_class)

        decoder_class = cls.get_decoder_class(args.decoder)
        decoder_conf = get_default_kwargs(decoder_class)

        ctc_conf = get_default_kwargs(CTC)
        e2e_conf = get_default_kwargs(ASRE2E)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        frontend_conf.update(config["frontend_conf"])
        normalize_conf.update(config["normalize_conf"])
        encoder_conf.update(config["encoder_conf"])
        decoder_conf.update(config["decoder_conf"])
        ctc_conf.update(config["ctc_conf"])
        e2e_conf.update(config["e2e_conf"])

        # 5. Reassign them to the configuration
        config.update(
            frontend_conf=frontend_conf,
            normalize_conf=normalize_conf,
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            e2e_conf=e2e_conf,
        )

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
    def frontend_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ("default", None)
        return choices

    @classmethod
    def get_frontend_class(cls, name: str) -> Type[AbsFrontend]:
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == "default":
            retval = DefaultFrontend
        else:
            raise RuntimeError(
                f"--frontend must be one of "
                f"{cls.frontend_choices()}: --frontend {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def normalize_choices(cls) -> Tuple[Optional[str], ...]:
        choices = ("global_mvn", "utterance_mvn", None)
        return choices

    @classmethod
    def get_normalize_class(cls, name: str) -> Type[AbsNormalize]:
        assert check_argument_types()
        if name.lower() == "global_mvn":
            retval = GlobalMVN
        elif name.lower() == "utterance_mvn":
            retval = UtteranceMVN
        else:
            raise RuntimeError(
                f"--normalize must be one of "
                f"{cls.normalize_choices()}: --normalize {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def encoder_choices(cls) -> Tuple[str, ...]:
        choices = ("transformer", "vgg_rnn", "rnn")
        return choices

    @classmethod
    def get_encoder_class(cls, name: str) -> Type[AbsEncoder]:
        assert check_argument_types()
        if name.lower() == "transformer":
            retval = TransformerEncoder
        elif name.lower() == "vgg_rnn":
            retval = VGGRNNEncoder
        elif name.lower() == "rnn":
            retval = RNNEncoder
        else:
            raise RuntimeError(
                f"--normalize must be one of "
                f"{cls.normalize_choices()}: --normalize {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def decoder_choices(cls) -> Tuple[str, ...]:
        choices = ("transformer", "rnn")
        return choices

    @classmethod
    def get_decoder_class(cls, name: str) -> Type[AbsDecoder]:
        assert check_argument_types()
        if name.lower() == "transformer":
            retval = TransformerDecoder
        elif name.lower() == "rnn":
            retval = RNNDecoder
        else:
            raise RuntimeError(
                f"--normalize must be one of "
                f"{cls.normalize_choices()}: --normalize {name}"
            )
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_collate_fn(
        cls, args: argparse.Namespace
    ) -> Callable[[Collection[Tuple[str, Dict[str, np.ndarray]]]],
                  Tuple[List[str], Dict[str, torch.Tensor]]]:
        assert check_argument_types()
        # NOTE(kamo): int value = 0 is reserved by CTC-blank symbol
        return CommonCollateFn(float_pad_value=0.0, int_pad_value=-1)

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
                bpemodel=args.bpemodel)
        else:
            retval = None
        assert check_return_type(retval)
        return retval

    @classmethod
    def required_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        if train:
            retval = ("speech", "text")
        else:
            # Recognition mode
            retval = ("speech",)
        return retval

    @classmethod
    def optional_data_names(cls, train: bool = True) -> Tuple[str, ...]:
        retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ASRE2E:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size }")

        # 1. frontend
        if args.input_size is None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            if hasattr(args, "frontend"):
                # Either one of frontend and input_size can be selected
                delattr(args, "frontend")
            args.frontend_conf = {}
            frontend = None
            input_size = args.input_size

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. Encoder
        encoder_class = cls.get_encoder_class(args.encoder)
        encoder = encoder_class(input_size=input_size, **args.encoder_conf)

        # 4. Decoder
        decoder_class = cls.get_decoder_class(args.decoder)

        decoder = decoder_class(
            vocab_size=vocab_size,
            encoder_output_size=encoder.output_size(),
            **args.decoder_conf,
        )

        # 4. CTC
        ctc = CTC(
            odim=vocab_size, encoder_output_sizse=encoder.output_size(), **args.ctc_conf
        )

        # 5. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 6. Build model
        model = ASRE2E(
            vocab_size=vocab_size,
            frontend=frontend,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.e2e_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 7. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

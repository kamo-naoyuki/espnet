import argparse
import logging
from typing import Any, Dict, Type, Tuple, Optional

import configargparse
from typeguard import check_argument_types, check_return_type

from espnet2.asr.controller import ASRModelController
from espnet2.asr.ctc import CTC
from espnet2.asr.encoder_decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder_decoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.frontend.default import DefaultFrontend
from espnet2.asr.normalize.abs_normalization import AbsNormalization
from espnet2.asr.normalize.global_mvn import GlobalMVN
from espnet2.asr.normalize.utterance_mvn import UtteranceMVN
from espnet2.tasks.abs_task import AbsTask
from espnet2.utils.get_default_kwargs import get_defaut_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import str_or_none, int_or_none


class ASRTask(AbsTask):
    @classmethod
    def add_arguments(cls, parser: configargparse.ArgumentParser = None) \
            -> configargparse.ArgumentParser:
        assert check_argument_types()
        # NOTE(kamo): Use '_' instead of '-' to avoid confusion
        if parser is None:
            parser = configargparse.ArgumentParser(
                description='Train ASR',
                config_file_parser_class=configargparse.YAMLConfigFileParser,
                formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

        AbsTask.add_arguments(parser)
        group = parser.add_argument_group(description='Task related')

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default('required')
        required += ['token_list']

        group.add_argument('--token_list', type=str_or_none, default=None,
                           help='A text mapping int-id to token')

        excl = group.add_mutually_exclusive_group()
        excl.add_argument('--idim', type=int_or_none, default=None,
                          help='The number of input dimension of the feature')
        excl.add_argument(
            '--frontend', type=str_or_none, default='default',
            choices=cls.frontend_choices(), help='Specify frontend class')
        group.add_argument(
            '--frontend_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for frontend class.')
        group.add_argument(
            '--normalize', type=str_or_none, default='utterance_mvn',
            choices=cls.normalize_choices(),
            help='Specify normalization class')
        group.add_argument(
            '--normalize_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for normalization class.')

        group.add_argument(
            '--encoder_decoder', type=str, default='rnn',
            choices=cls.encoder_decoder_choices(),
            help='Specify Encoder-Decoder type')
        group.add_argument(
            '--encoder_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Encoder class.')
        group.add_argument(
            '--decoder_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for Decoder class.')
        group.add_argument(
            '--ctc_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for CTC class.')
        group.add_argument(
            '--model_conf', action=NestedDictAction, default=dict(),
            help='The keyword arguments for ModelController class.')

        return parser

    @classmethod
    def exclude_opts(cls) -> Tuple[str, ...]:
        """The options not to be shown by --print_config"""
        assert check_argument_types()
        return AbsTask.exclude_opts()

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        assert check_argument_types()
        # This method is used only for --print_config

        # 0. Parse command line arguments
        parser = ASRTask.add_arguments()
        args, _ = parser.parse_known_args()

        # 1. Get the default values from class.__init__
        if args.idim is None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend_conf = get_defaut_kwargs(frontend_class)
        else:
            if hasattr(args, 'frontend'):
                # Either one of frontend and idim can be selected
                delattr(args, 'frontend')
            frontend_conf = {}
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize_conf = get_defaut_kwargs(normalize_class)
        else:
            normalize_conf = None

        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder_conf = get_defaut_kwargs(encoder_class)
        decoder_conf = get_defaut_kwargs(decoder_class)
        ctc_conf = get_defaut_kwargs(CTC)
        model_conf = get_defaut_kwargs(ASRModelController)

        # 2. Create configuration-dict from command-arguments
        config = vars(args)

        # 3. Update the dict using the inherited configuration from BaseTask
        config.update(AbsTask.get_default_config())

        # 4. Overwrite the default config by the command-arguments
        frontend_conf.update(config['frontend_conf'])
        normalize_conf.update(config['normalize_conf'])
        encoder_conf.update(config['encoder_conf'])
        decoder_conf.update(config['decoder_conf'])
        ctc_conf.update(config['ctc_conf'])

        # 5. Reassign them to the configuration
        config.update(
            frontend_conf=frontend_conf,
            normalize_conf=normalize_conf,
            encoder_conf=encoder_conf,
            decoder_conf=decoder_conf,
            ctc_conf=ctc_conf,
            model_conf=model_conf)

        # 6. Excludes the options not to be shown
        for k in cls.exclude_opts():
            config.pop(k)

        assert check_return_type(config)
        return config

    @classmethod
    def frontend_choices(cls) -> Tuple[Optional[str], ...]:
        assert check_argument_types()
        choices = ('default',)
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_frontend_class(cls, name: str) -> Type[AbsFrontend]:
        assert check_argument_types()
        # NOTE(kamo): Don't use getattr or dynamic_import
        # for readability and debuggability as possible
        if name.lower() == 'default':
            retval = DefaultFrontend
        else:
            raise RuntimeError(
                f'--frontend must be one of '
                f'{cls.frontend_choices()}: --frontend {name}')
        assert check_return_type(retval)
        return retval

    @classmethod
    def normalize_choices(cls) -> Tuple[Optional[str], ...]:
        assert check_argument_types()
        choices = ('global_mvn', 'utterance_mvn')
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        choices += (None,)
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_normalize_class(cls, name: str) -> Type[AbsNormalization]:
        assert check_argument_types()
        if name.lower() == 'global_mvn':
            retval = GlobalMVN
        elif name.lower() == 'utterance_mvn':
            retval = UtteranceMVN
        else:
            raise RuntimeError(
                f'--normalize must be one of '
                f'{cls.normalize_choices()}: --normalize {name}')
        assert check_return_type(retval)
        return retval

    @classmethod
    def encoder_decoder_choices(cls) -> Tuple[str, ...]:
        assert check_argument_types()
        choices = ('Transformer', 'rnn')
        choices += tuple(x.lower() for x in choices if x != x.lower()) \
            + tuple(x.upper() for x in choices if x != x.upper())
        assert check_return_type(choices)
        return choices

    @classmethod
    def get_encoder_decoder_class(cls, name: str) \
            -> Tuple[Type[AbsEncoder], Type[AbsDecoder]]:
        assert check_argument_types()
        if name.lower() == 'transformer':
            from espnet2.asr.encoder_decoder.transformer.encoder import Encoder
            from espnet2.asr.encoder_decoder.transformer.decoder import Decoder
            retval = Encoder, Decoder

        elif name.lower() == 'rnn':
            from espnet2.asr.encoder_decoder.rnn.decoder import Decoder
            from espnet2.asr.encoder_decoder.rnn.encoder import Encoder
            retval = Encoder, Decoder

        else:
            raise RuntimeError(
                f'--encoder_decoder must be one of '
                f'{cls.encoder_decoder_choices()}: --encoder_decoder {name}')
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ASRModelController:
        assert check_argument_types()
        if isinstance(args.token_list, str):
            with open(args.token_list) as f:
                token_list = [line.rstrip() for line in f]

            # Overwriting token_list to keep it as "portable".
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError('token_list must be str or list')
        vocab_size = len(token_list)
        logging.info(f'Vocabulary size: {vocab_size }')

        # 1. frontend
        if args.idim is None:
            frontend_class = cls.get_frontend_class(args.frontend)
            frontend = frontend_class(**args.frontend_conf)
            idim = frontend.out_dim()
        else:
            if hasattr(args, 'frontend'):
                # Either one of frontend and idim can be selected
                delattr(args, 'frontend')
            args.frontend_conf = {}
            frontend = None
            idim = args.idim

        # 2. Normalization layer
        if args.normalize is not None:
            normalize_class = cls.get_normalize_class(args.normalize)
            normalize = normalize_class(**args.normalize_conf)
        else:
            normalize = None

        # 3. Encoder, Decoder
        encoder_class, decoder_class = \
            cls.get_encoder_decoder_class(args.encoder_decoder)
        encoder = encoder_class(idim=idim,
                                **args.encoder_conf)
        decoder = decoder_class(odim=vocab_size,
                                encoder_out_dim=encoder.out_dim(),
                                **args.decoder_conf)

        # 4. CTC
        ctc = CTC(odim=vocab_size, encoder_out_dim=encoder.out_dim(),
                  **args.ctc_conf)

        # 5. RNN-T Decoder (Not implemented)
        rnnt_decoder = None

        # 6. Build controller
        model = ASRModelController(
            vocab_size=vocab_size,
            frontend=frontend,
            normalize=normalize,
            encoder=encoder,
            decoder=decoder,
            ctc=ctc,
            rnnt_decoder=rnnt_decoder,
            token_list=token_list,
            **args.model_conf)

        assert check_return_type(model)
        return model
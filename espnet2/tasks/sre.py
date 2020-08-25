import argparse
import logging
from typing import Tuple
import warnings

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.global_mvn import GlobalMVN
from espnet2.layers.instance_norm import ESPnetInstanceNorm1d
from espnet2.layers.utterance_mvn import UtteranceMVN
from espnet2.sre.data.pairwise_dataset import PairwiseDataset
from espnet2.sre.data.pairwise_batch_sampler import PairwiseBatchSampler
from espnet2.sre.data.chunk_preprocessor import FeatsExtractChunkPreprocessor
from espnet2.sre.espnet_model import ESPnetSREModel
from espnet2.sre.loss.aam_softmax_loss import AAMSoftmaxLoss
from espnet2.sre.loss.abs_loss import AbsLoss
from espnet2.sre.loss.am_softmax_loss import AMSoftmaxLoss
from espnet2.sre.loss.angleproto_loss import AngleProtoLoss
from espnet2.sre.loss.ge2e_loss import GE2ELoss
from espnet2.sre.loss.pairwise_loss import PairwiseLoss
from espnet2.sre.loss.proto_loss import ProtoLoss
from espnet2.sre.loss.softmax_loss import SoftmaxLoss
from espnet2.sre.net.abs_net import AbsNet
from espnet2.sre.net.resnet34 import ResNet34
from espnet2.sre.net.vggvox import VGGVox
from espnet2.sre.pooling.abs_pooling import AbsPooling
from espnet2.sre.pooling.global_average_pooling import GlobalAveragePooling
from espnet2.sre.pooling.global_max_pooling import GlobalMaxPooling
from espnet2.sre.pooling.self_attention_pooling import SelfAttentionPooling
from espnet2.tasks.abs_task import AbsTask
from espnet2.tasks.abs_task import IteratorOptions
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices
from espnet2.train.collate_fn import CommonCollateFn
from espnet2.train.trainer import Trainer
from espnet2.utils.get_default_kwargs import get_default_kwargs
from espnet2.utils.nested_dict_action import NestedDictAction
from espnet2.utils.types import humanfriendly_parse_size_or_none
from espnet2.utils.types import str_or_none


net_choices = ClassChoices(
    "net",
    classes=dict(vggvox=VGGVox, resnet=ResNet34),
    type_check=AbsNet,
    default="vggvox",
)
normalize_choices = ClassChoices(
    "normalize",
    classes=dict(
        global_mvn=GlobalMVN,
        utterance_mvn=UtteranceMVN,
        instance_norm=ESPnetInstanceNorm1d,
    ),
    type_check=AbsNormalize,
    default="instance_norm",
    optional=True,
)
pooling_choices = ClassChoices(
    "pooling",
    classes=dict(
        max=GlobalMaxPooling, tap=GlobalAveragePooling, sap=SelfAttentionPooling,
    ),
    type_check=AbsPooling,
    default="sap",
)
loss_choices = ClassChoices(
    "loss",
    classes=dict(
        softmax=SoftmaxLoss,
        am_softmax=AMSoftmaxLoss,
        aam_softmax=AAMSoftmaxLoss,
        ge2e_softmax=GE2ELoss,
        pairwise_loss=PairwiseLoss,
        proto_loss=ProtoLoss,
        angle_proto_loss=AngleProtoLoss,
    ),
    type_check=AbsLoss,
    default="softmax",
)


class SRETask(AbsTask):
    # If you need more than one optimizers, change this value
    num_optimizers: int = 1

    # Add variable objects configurations
    class_choices_list = [
        # --normalize and --normalize_conf
        normalize_choices,
        # --net and --net_conf
        net_choices,
        # --loss and --loss_conf
        loss_choices,
        # --pooling and --pooling_conf
        pooling_choices,
    ]

    # If you need to modify train() or eval() procedures, change Trainer class here
    trainer = Trainer

    @classmethod
    def add_task_arguments(cls, parser: argparse.ArgumentParser):
        parser.set_defaults(
            iterator_type="task", num_att_plot=0, valid_batch_type="unsorted"
        )
        group = parser.add_argument_group(description="Task related")

        # NOTE(kamo): add_arguments(..., required=True) can't be used
        # to provide --print_config mode. Instead of it, do as
        required = parser.get_default("required")
        required += ["utt2spk", "fs"]

        group.add_argument(
            "--utt2spk",
            type=str,
            default=None,
            help="A text file mapping utterance id to speaker id",
        )

        group.add_argument(
            "--init",
            type=lambda x: str_or_none(x.lower()),
            default=None,
            help="The initialization method",
            choices=[
                "chainer",
                "xavier_uniform",
                "xavier_normal",
                "kaiming_uniform",
                "kaiming_normal",
                None,
            ],
        )

        group.add_argument(
            "--num_pair", type=int, default=1, help="The number samplers per pair",
        )
        group.add_argument(
            "--embed_size",
            type=int,
            default=512,
            help="The size of the output embedding vector",
        )
        group.add_argument(
            "--model_conf",
            action=NestedDictAction,
            default=get_default_kwargs(ESPnetSREModel),
            help="The keyword arguments for model class.",
        )

        group = parser.add_argument_group(description="Preprocess related")
        group.add_argument(
            "--fs",
            type=humanfriendly_parse_size_or_none,
            default=None,
            help="Sampling frequency",
        )
        group.add_argument(
            "--preprocess_conf",
            action=NestedDictAction,
            default=get_default_kwargs(FeatsExtractChunkPreprocessor),
            help="Preprocess conf",
        )
        for class_choices in cls.class_choices_list:
            # Append --<name> and --<name>_conf.
            # e.g. --encoder and --encoder_conf
            class_choices.add_arguments(group)

    @classmethod
    def build_task_iter_factory(
        cls, args: argparse.Namespace, iter_options: IteratorOptions, mode: str
    ):
        # NOTE(kamo):
        # For training: Use pairwise chunked utterances
        # For validation in training and inference mode: Use per utterances

        if iter_options.train:
            if len(iter_options.shape_files) != 0:
                raise RuntimeError("shape file is not used for this task")

            batch_sampler = PairwiseBatchSampler(
                batch_size=args.batch_size,
                key_file=iter_options.data_path_and_name_and_type[0][0],
                utt2spk=args.utt2spk,
                distributed=iter_options.distributed,
                num_pair=args.num_pair,
                shuffle=iter_options.train,
            )

            dataset = PairwiseDataset(
                iter_options.data_path_and_name_and_type,
                float_dtype=args.train_dtype,
                preprocess=iter_options.preprocess_fn,
                max_cache_size=iter_options.max_cache_size,
            )
            cls.check_task_requirements(
                dataset, args.allow_variable_data_keys, train=iter_options.train
            )

            if iter_options.num_batches is not None:
                raise NotImplementedError

            logging.info(f"[{mode}] dataset:\n{dataset}")

            return SequenceIterFactory(
                dataset=dataset,
                batches=batch_sampler,
                seed=args.seed,
                num_iters_per_epoch=iter_options.num_iters_per_epoch,
                shuffle=iter_options.train,
                num_workers=args.num_workers,
                collate_fn=iter_options.collate_fn,
                pin_memory=args.ngpu > 0,
            )
        else:
            if iter_options.batch_type != "unsorted":
                warnings.warn("Batch type should be unsorted")

            return cls.build_sequence_iter_factory(args, iter_options, mode)

    @classmethod
    def build_collate_fn(cls, args: argparse.Namespace, train: bool):
        assert check_argument_types()
        if train:
            # Use default_collate because we handle only fixed length chunk in this task
            return torch.utils.data.dataloader.default_collate
        else:
            # For validation or collect stats mode
            return CommonCollateFn(
                float_pad_value=0.0, int_pad_value=0, not_sequence=["label"]
            )

    @classmethod
    def build_preprocess_fn(cls, args: argparse.Namespace, train: bool):
        assert check_argument_types()
        if train:
            return FeatsExtractChunkPreprocessor(
                train=train, utt2spk=args.utt2spk, fs=args.fs, **args.preprocess_conf
            )
        else:

            # Disable chunking for not training mode
            conf = args.preprocess_conf.copy()
            conf.update(cut_chunk=False)

            return FeatsExtractChunkPreprocessor(
                train=train, utt2spk=args.utt2spk, fs=args.fs, **args.preprocess_conf
            )

    @classmethod
    def required_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        assert check_argument_types()
        if inference:
            retval = ("speech",)
        elif train:
            retval = ("speech",)
        else:
            # For validation or collect stats mode
            retval = ("speech", "reference", "label")
        assert check_return_type(retval)
        return retval

    @classmethod
    def optional_data_names(
        cls, train: bool = True, inference: bool = False
    ) -> Tuple[str, ...]:
        assert check_argument_types()
        if inference:
            retval = ("reference", "label")
        else:
            retval = ()
        assert check_return_type(retval)
        return retval

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetSREModel:
        assert check_argument_types()

        # train is False because we preprocess_fn only to get num_features here
        preprocess_fn = cls.build_preprocess_fn(args, train=False)

        # 1. Normalization layer
        if args.normalize is not None:
            normalize_class = normalize_choices.get_class(args.normalize)
            conf = args.normalize_conf
            if issubclass(normalize_class, ESPnetInstanceNorm1d):
                conf["num_features"] = preprocess_fn.get_num_features()
            normalize = normalize_class(**conf)

        else:
            normalize = None

        # 2. Build internal network
        net_class = net_choices.get_class(args.net)
        net = net_class(input_size=preprocess_fn.get_num_features(), **args.net_conf)

        # 3. Build internal network
        pooling_class = pooling_choices.get_class(args.pooling)
        if issubclass(pooling_class, SelfAttentionPooling):
            pooling = pooling_class(input_size=net.output_size(), **args.pooling_conf)
        else:
            pooling = pooling_class(**args.pooling_conf)

        # 4. Build linear
        linear = torch.nn.Linear(net.output_size(), args.embed_size)

        # 5. Build loss function
        loss_class = loss_choices.get_class(args.loss)
        if issubclass(loss_class, (SoftmaxLoss, AMSoftmaxLoss, AAMSoftmaxLoss)):
            loss = loss_class(
                input_size=args.embed_size,
                num_classes=preprocess_fn.get_num_spk(),
                **args.loss_conf,
            )
        else:
            loss = loss_class(**args.loss_conf)

        # 6. Build model
        model = ESPnetSREModel(
            normalize=normalize,
            net=net,
            pooling=pooling,
            linear=linear,
            loss=loss,
            **args.model_conf,
        )

        # FIXME(kamo): Should be done in model?
        # 7. Initialize
        if args.init is not None:
            initialize(model, args.init)

        assert check_return_type(model)
        return model

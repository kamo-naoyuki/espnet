from pathlib import Path
from typing import Union

import numpy as np
import torch

from espnet2.samplers.abs_sampler import AbsSampler
from espnet2.sre.utils import read_utt2spk


class PairwiseBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_size: int,
        key_file: Union[str, Path],
        utt2spk: Union[str, Path],
        num_pair: int,
        shuffle: bool = False,
        distributed: bool = False,
    ):
        self.num_pair = num_pair
        self.distributed = distributed
        self.shuffle = shuffle
        self.batch_size = batch_size

        # NOTE(kamo): The utts in utt2spk can be superset of key_file.
        #   e.g. The union of training set and validation set
        self.utt2spk_dict, spk2utt_dict, _, _, = read_utt2spk(utt2spk)
        self.num_spk = len(spk2utt_dict)

        self.spk2utt_dict_closed = {}
        with open(key_file, encoding="utf-8") as f:
            self.utt_list = []
            for line in f:
                utt, *spk = line.split(maxsplit=1)
                if utt not in self.utt2spk_dict:
                    raise RuntimeError(f"Unknown key: {utt}")
                self.utt_list.append(utt)
                spk = self.utt2spk_dict[utt]
                self.spk2utt_dict_closed.setdefault(spk, []).append(utt)
        if len(self.utt_list) == 0:
            raise RuntimeError(f"Empty file: {key_file}")
        if len(self.spk2utt_dict_closed) < self.num_pair:
            raise RuntimeError(
                f"Too few speakers: {len(self.spk2utt_dict_closed)} <= {self.num_pair}"
            )
        # NOTE(kamo): Always drop last if not divisible if num_batches != 1
        #   (At least 1 number of batches here)
        self.num_batches = max(len(self.utt_list) // self.batch_size, 1)
        if self.num_batches == 1:
            self.batch_size = len(self.utt_list)

        if self.distributed:
            ws = torch.distributed.get_world_size()
            if ws > batch_size:
                raise RuntimeError(
                    "World size is larger than batch_size: " f"{ws} > {batch_size}"
                )
            if ws > len(self.utt_list):
                raise RuntimeError(
                    "World size is larger than the number of utterances: "
                    f"{ws} > {len(self.utt_list)}"
                )

    def __repr__(self):
        _ret = f"{self.__class__.__name__}("
        _ret += f"num_spk={self.num_spk}, "
        _ret += f"num_spk_closed={len(self.spk2utt_dict_closed)}, "
        _ret += f"num_utts={len(self.utt_list)}, "
        _ret += f"batch_size={self.batch_size}, "
        _ret += f"num_batches={self.num_batches})"
        return _ret

    def generate(self, seed: int = None):
        state = np.random.RandomState(seed)

        key_list = []
        for utt in self.utt_list:
            keys = [utt]
            spk = self.utt2spk_dict[utt]

            if self.shuffle:
                spks = state.choice(
                    list(set(self.spk2utt_dict_closed) - {spk}),
                    self.num_pair - 1,
                    replace=False,
                )
            else:
                spks = list(set(self.spk2utt_dict_closed) - {spk})[: self.num_pair - 1]

            for spk2 in spks:
                utts = self.spk2utt_dict_closed[spk2]
                if self.shuffle:
                    utt2 = state.choice(utts, 1)[0]
                else:
                    utt2 = utts[0]
                keys.append(utt2)
            key_list.append(tuple(keys))

        # NOTE(kamo): Always drop last if not divisible
        batches = [
            tuple(key_list[i * self.batch_size : (i + 1) * self.batch_size])
            for i in range(self.num_batches)
        ]
        if self.distributed:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            batches = [batch[rank::world_size] for batch in batches]
        return batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # Don't use for this task
        raise NotImplementedError

#!/usr/bin/env python3
import argparse
import collections
from distutils.util import strtobool
from io import StringIO
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from espnet.utils.cli_utils import get_commandline_args
from espnet2.train.dataset import DATA_TYPES
from espnet2.train.dataset import ESPnetDataset
from espnet2.utils.hdf5_corpus import H5FileWrapper
from espnet2.utils.types import str2triple_str


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate a HDF5 corpus from scp files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        action="append",
        default=[],
    )
    parser.add_argument("--shape_file", type=str, action="append", default=[])
    parser.add_argument("--out", type=str, help="Output HDF5 file name", required=True)
    parser.add_argument(
        "--append", type=strtobool, default=False, help="Open with append mode"
    )
    parser.add_argument(
        "--num_samples_per_group",
        type=int,
        default=400000,
        help="The maximum number of samples per group."
    )
    return parser


def read_scp(data_path):
    with open(data_path) as f:
        for linenum, line in enumerate(f):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) != 2:
                raise RuntimeError(
                    f"must have two or more columns: " f"{line}({data_path}:{linenum})"
                )
            k, v = sps
            yield linenum, k, v


def main(cmd=None):
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = get_parser()
    args = parser.parse_args(cmd)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    """Generate 'HDF5 corpus'

    ESPnet2 supports two types of methods for data inputting:
      1. Separated files like feats.scp, text, etc.
      2. A HDF5 file created by combining them

    The HDF5 must have the following structure e.g.:
      - speech/type="sound"
      - speech/data
          - group0/
              - id1="/some/where/a.wav"
              - id2="/some/where/b.wav"
              - ...
          - group1/
              - id100="/some/where/foo.wav"
              - id101="/some/where/bar.wav"
              - ...
          - ...
      - text/type="direct"
      - text/data
          - group0/
              - id1="abc def"
              - id2="hello world"
              - ...
          - group1/
              - id100="foo bar"
              - id101="aaa bbb"
              - ...
          - ...
      - shape_files/0
          - group0/
              - id1=(10000,)
              - id2=(14000,)
              - ...
          - group1/
              - id100=(10000,)
              - id101=(14000,)
              - ...
          - ...
      - shape_files/1
          - group0/
              - id1=(2,)
              - id2=(2,)
              - ...
          - group1/
              - id100=(2,)
              - id101=(2,)
              - ...
          - ...
    """
    with h5py.File(args.out, "w+" if args.append else "w") as fout:
        for data_path, name, type in args.data_path_and_name_and_type:
            if type not in DATA_TYPES:
                raise RuntimeError(f"Must be one of {list(DATA_TYPES)}: {type}")
            grp = fout.create_group(f"{name}/data/")

            # If scp file, insert the reference file path instead of ndarray
            # e.g. uttid_A /some/where/a.wav
            # => f["name/data/uttid_A"] = "/some/where/a.wav"
            if type in ("sound", "npy", "kaldi_ark", "pipe_wav"):
                fout[f"{name}/type"] = type
                for linenum, k, v in tqdm(read_scp(data_path), desc=data_path):
                    if linenum % args.num_samples_per_group == 0:
                        num = linenum // args.num_samples_per_group
                        subgrp = grp.create_group(f"group{num}")
                    subgrp[k] = v

            # The other case, set ndarray/str directly
            else:
                fout[f"{name}/type"] = "direct"
                loader = ESPnetDataset.build_loader(data_path, type)
                if isinstance(loader, collections.abc.Mapping):
                    for linenum, k in tqdm(enumerate(loader), desc=data_path):
                        if linenum % args.num_samples_per_group == 0:
                            num = linenum // args.num_samples_per_group
                            subgrp = grp.create_group(f"group{num}")
                        subgrp[k] = loader[k]
                elif isinstance(loader, collections.abc.Collection):
                    for linenum, v in tqdm(enumerate(loader), desc=data_path):
                        if linenum % args.num_samples_per_group == 0:
                            num = linenum // args.num_samples_per_group
                            subgrp = grp.create_group(f"group{num}")
                        subgrp[k] = v
                else:
                    raise RuntimeError(f"{type} is not supported")

        for idx, shape_file in enumerate(args.shape_file):
            grp = fout.create_group(f"shape_files/{idx}")
            for linenum, k, v in tqdm(read_scp(shape_file), desc=shape_file):
                if linenum % args.num_samples_per_group == 0:
                    num = linenum // args.num_samples_per_group
                    subgrp = grp.create_group(f"group{num}")
                v = np.loadtxt(StringIO(v), ndmin=1, dtype=np.long, delimiter=",")
                subgrp[k] = v

        # Check having same keys set
        first_group = H5FileWrapper(
            fout[args.data_path_and_name_and_type[0][1]]["data"]
        )
        for data_path, name, type in args.data_path_and_name_and_type:
            group = H5FileWrapper(fout[name]["data"])
            if len(first_group) != len(group):
                raise RuntimeError(
                    f"Keys are mismatched between "
                    f"{args.data_path_and_name_and_type[0][0]} and {data_path}"
                )
            for k in first_group:
                if k not in group:
                    raise RuntimeError(
                        f"Keys are mismatched between "
                        f"{args.data_path_and_name_and_type[0][0]} and {data_path}"
                    )

        for idx, shape_file in enumerate(args.shape_file):
            group = H5FileWrapper(fout["shape_files"][str(idx)])
            if len(first_group) != len(group):
                raise RuntimeError(
                    f"Keys are mismatched between "
                    f"{args.data_path_and_name_and_type[0][0]} and {shape_file}"
                )
            for k in first_group:
                if k not in group:
                    raise RuntimeError(
                        f"Keys are mismatched between "
                        f"{args.data_path_and_name_and_type[0][0]} and {shape_file}"
                    )


if __name__ == "__main__":
    main()

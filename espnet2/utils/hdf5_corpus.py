import collections
import itertools
from pathlib import Path
from typing import Union

import h5py
from typeguard import check_argument_types


def is_hdf5_corpus_format(file: Union[str, Path]) -> bool:
    """Check format of HDF5.

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
    with h5py.File(file, "r") as f:
        keys = [k for k in f if k != "shape_files"]

        for key in keys:
            if "type" not in f[key] or "data" not in f[key]:
                return False
        return "shape_files/0" in f


class H5FileWrapper(collections.abc.Mapping):
    def __init__(self, file: h5py.Group):
        assert check_argument_types()
        self.file = file

    def __repr__(self) -> str:
        return repr(self.file)

    def __getitem__(self, item):
        for g in self.file.values():
            if item in g:
                return g[item][()]
        else:
            raise KeyError(item)

    def __len__(self) -> int:
        return sum(len(g) for g in self.file.values())

    def __iter__(self):
        return itertools.chain(*self.file.values())

    def __contains__(self, item) -> bool:
        for g in self.file.values():
            if item in g:
                return True
        else:
            return False

    def keys(self):
        return list(itertools.chain(*(g.keys() for g in self.file.values())))

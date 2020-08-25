from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.train.dataset import AbsDataset
from espnet2.train.dataset import ESPnetDataset


class PairwiseDataset(AbsDataset):
    def __init__(
        self,
        path_name_type_list: Collection[Tuple[str, str, str]],
        preprocess: Callable[
            [str, Dict[str, np.ndarray]], Dict[str, np.ndarray]
        ] = None,
        float_dtype: str = "float32",
        int_dtype: str = "long",
        max_cache_size: Union[float, int, str] = 0.0,
    ):
        assert check_argument_types()
        self.dataset = ESPnetDataset(
            path_name_type_list=path_name_type_list,
            preprocess=preprocess,
            float_dtype=float_dtype,
            int_dtype=int_dtype,
            max_cache_size=max_cache_size,
        )

    def has_name(self, name) -> bool:
        return self.dataset.has_name(name)

    def names(self) -> Tuple[str, ...]:
        return self.dataset.names()

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.dataset})"

    def __getitem__(
        self, uid: Union[List[str], Tuple[str, ...]]
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:
        assert check_argument_types()

        data = {}
        # 1. A pair of uid is given as list of str here
        # and derive the data from each uid
        for uid_ in uid:
            _, data_ = self.dataset[uid_]
            for key_, value_ in data_.items():
                data.setdefault(key_, []).append(value_)

        # 2. Concatenate the pair of numpy arrays
        # Note that all data must have same length,
        # thus fixed length chunks should be cut
        # from each utterances by the preprocessor in advance.
        concat_data = {}
        for key, value in data.items():
            # a list as N x (Dim, ...) -> an array as (N, Dim, ...)
            # where N is number of utts in a pair (this is not batch-size)
            concat_data[key] = np.stack(value, axis=0)
        return uid, concat_data

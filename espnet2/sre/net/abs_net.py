from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import torch


class AbsNet(torch.nn.Module, ABC):
    @abstractmethod
    def output_size(self):
        raise NotImplementedError

    @abstractmethod
    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, O)
        """
        raise NotImplementedError

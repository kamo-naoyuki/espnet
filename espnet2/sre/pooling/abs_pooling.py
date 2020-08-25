from abc import ABC
from abc import abstractmethod
from typing import Tuple

import torch


class AbsPooling(torch.nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            x_lengths: (B,)
        Returns:
            (B, D)
        """
        raise NotImplementedError

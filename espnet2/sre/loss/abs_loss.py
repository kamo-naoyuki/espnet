from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import Tuple

import torch


class AbsLoss(torch.nn.Module, ABC):
    @abstractmethod
    def forward(
        self, x: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError

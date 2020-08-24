"""
Adapted from https://github.com/clovaai/voxceleb_trainer
(MIT License)
"""
from typing import List

import torch


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1,)
) -> List[torch.Tensor]:
    """Computes the precision@k for the specified values of k"""
    assert output.size()[:-1] == target.size(), (output.size(), target.size())
    if output.dim() >= 3:
        output = output.view(-1, output.size(-1))
        target = target.view(-1)

    maxk = max(topk)
    batch_size = torch.prod(torch.tensor(target.size()))

    # output: (B, Nskp) -> pred: (B, maxk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target[:, None])

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k / batch_size)
    return res

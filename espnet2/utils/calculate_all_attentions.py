from collections import defaultdict
from typing import Dict, List

import torch

from espnet.nets.pytorch_backend.transformer.attention import \
    MultiHeadedAttention


@torch.no_grad()
def calculate_all_attentions(model: torch.nn.Module, **batch: torch.Tensor) \
        -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    # TODO(kamo): Implement the base class representing Attention
    #  Now only MultiHeadedAttention is supported

    # 1. Derive the key names e.g. input, output
    keys = []
    for k in batch:
        if k + '_lengths' in batch:
            keys.append(k)
    bs = len(batch[keys[0]])
    assert all(len(v) == bs for v in batch.values()), \
        {k: v.shape for k, v in batch.items()}

    # 2. Register forward_hook fn to save the output from specific layers
    outputs = {}
    handles = {}
    for name, modu in model.named_modules():
        def hook(module, input, output, name=name):
            outputs[name] = output

        if isinstance(modu, MultiHeadedAttention):
            handle = modu.register_forward_hook(hook)
            handles[name] = handle

    # 3. Just forward one by one sample.
    # Batch-mode can't be used to keep small requirements for each models.
    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = \
            {k: batch[k][ibatch, None, :batch[k + '_lengths'][ibatch]]
             for k in keys}

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {k + '_lengths': batch[k + '_lengths'][ibatch, None]
             for k in keys})

        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            return_dict[name].append(output)

    # 4. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return return_dict
# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


import torch.distributed as dist
import torch

import numpy as np
import torch
from torch import Tensor
from training.isomaxplus import IsoMaxPlusLossSecondPart
from training.mask_classification_loss import MaskClassificationLoss


class MaskClassificationLossIsomax(MaskClassificationLoss):
    def __init__(
        self,
        *args,
        **kargs
    ):
        super().__init__(*args, **kargs)
    
    def loss_labels(
        self, class_queries_logits: Tensor, class_labels: list[Tensor], indices: tuple[np.array]
    ) -> dict[str, Tensor]:
        
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        
        pred_logits_flat = pred_logits.view(-1, self.num_labels + 1) 
        target_classes_flat = target_classes.view(-1)
        
        weights = torch.ones_like(target_classes_flat, dtype=torch.float)
        weights[target_classes_flat == self.num_labels] = self.eos_coef

        criterion = IsoMaxPlusLossSecondPart()
        
        loss = criterion(pred_logits_flat, target_classes_flat, weights=weights)

        return {"loss_isomax": loss}

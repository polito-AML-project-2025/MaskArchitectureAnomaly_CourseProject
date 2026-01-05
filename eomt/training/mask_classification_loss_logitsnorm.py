# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch

import numpy as np
import torch
from torch import Tensor, nn
from training.mask_classification_loss import MaskClassificationLoss
import torch.nn.functional as F


class MaskClassificationLossLogitsnorm(MaskClassificationLoss):
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
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)  # shape of (batch_size, num_queries)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )  # shape of (batch_size, num_queries)
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels, dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o
        # Permute target_classes (batch_size, num_queries, num_labels) -> (batch_size, num_labels, num_queries)
        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(F.normalize(pred_logits_transposed, eps=1e-7), target_classes)
        losses = {"loss_cross_entropy_logitsnorm": loss_ce}
        return losses
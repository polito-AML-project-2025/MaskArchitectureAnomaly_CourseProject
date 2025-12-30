# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F

class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        rba_ood_supervision_enabled: bool = False,
        rba_coefficient:float = 1e-5,
        ood_label_id: int=254
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.rba_coefficient = rba_coefficient
        self.rba_ood_supervision_enabled = rba_ood_supervision_enabled
        self.ood_label_id = ood_label_id
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        
        if self.rba_ood_supervision_enabled:
            mask_labels = []
            class_labels = []
            ood_mask = []

            for target in targets:
                ind_indices = (target["labels"] != self.ood_label_id)
                ood_indices = (target["labels"] == self.ood_label_id)

                mask_labels.append(target["masks"][ind_indices].to(masks_queries_logits.dtype))
                class_labels.append(target["labels"][ind_indices].long())
                
                
                if ood_indices.any():
                    combined_mask = target["masks"][ood_indices].any(dim=0).to(masks_queries_logits.dtype)
                else:
                    h, w = target["masks"].shape[-2:]
                    combined_mask = torch.zeros((h, w), device=masks_queries_logits.device, dtype=masks_queries_logits.dtype)
                
                ood_mask.append(combined_mask)
        else:
            mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
            ]
            class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        if self.rba_ood_supervision_enabled:
            loss_rba = self.loss_rba(class_queries_logits, masks_queries_logits, ood_mask)
            return {**loss_masks, **loss_classes, **loss_rba}
        else:
            return {**loss_masks, **loss_classes}
    
    def loss_rba(self, class_queries_logits, masks_queries_logits, ood_mask):
        ood_mask = torch.stack(ood_mask).to(masks_queries_logits.device)

        if ood_mask.shape[-2:] != masks_queries_logits.shape[-2:]:
            ood_mask = F.interpolate(
                ood_mask.unsqueeze(1).float(),        
                size=masks_queries_logits.shape[-2:], 
                mode="nearest"                        
            ).squeeze(1)                              

        ood_mask = ood_mask > 0.5

        pixel_logits = torch.einsum(
            "bqhw, bqc -> bchw",
            masks_queries_logits.sigmoid(),
            class_queries_logits.softmax(dim=-1)[..., :-1],
        )
        alpha = 5
        rba_l = (torch.square(torch.clamp((alpha + pixel_logits.tanh().sum(dim=1)[ood_mask]), min=0))).sum()
        
        return {"loss_rba": rba_l}
        

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "loss_cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            elif "loss_rba" in loss_key:
                weighted_loss = loss * self.rba_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore

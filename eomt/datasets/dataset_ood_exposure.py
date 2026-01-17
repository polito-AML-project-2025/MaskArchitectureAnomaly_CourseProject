import random
import numpy as np
import torch
from torchvision import tv_tensors

from datasets.dataset import Dataset
from datasets.coco import COCOLoader

class CityscapesCocoOODDataset(Dataset):
    def __init__(
        self,
        *args,
        ood_prob: float = 0.5,
        ood_coco_root: str = None,
        ood_label_id: int = 254,
        ood_n_sample: int = 2000,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        print("ood_prob: ", ood_prob)
        
        self.ood_prob = ood_prob
        self.ood_label_id = ood_label_id
        
        print(ood_coco_root)
        self.coco_loader = COCOLoader(root=ood_coco_root, max_samples=ood_n_sample)

    @staticmethod
    def extract_bboxes(mask):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return 0, 0, 0, 0
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return y1, x1, y2 + 1, x2 + 1

    def inject_ood(self, base_img_tensor, base_target_dict, ood_img_np, ood_mask_np):
        img_np = base_img_tensor.permute(1, 2, 0).numpy().copy()
        h_img, w_img, _ = img_np.shape

        binary_ood_mask = (ood_mask_np > 0)
        
        y1, x1, y2, x2 = self.extract_bboxes(binary_ood_mask)
        if y2 <= y1 or x2 <= x1: 
            return base_img_tensor, base_target_dict 

        cut_object = ood_img_np[y1:y2, x1:x2]
        cut_mask = binary_ood_mask[y1:y2, x1:x2]
        
        h_obj, w_obj = cut_object.shape[:2]

        if h_img < h_obj or w_img < w_obj:
            return base_img_tensor, base_target_dict 
        
        h_start = random.randint(0, h_img - h_obj)
        w_start = random.randint(0, w_img - w_obj)
        h_end = h_start + h_obj
        w_end = w_start + w_obj

        roi = img_np[h_start:h_end, w_start:w_end]
        mask_3d = np.repeat(cut_mask[:, :, np.newaxis], 3, axis=2)
        roi[mask_3d] = cut_object[mask_3d]
        img_np[h_start:h_end, w_start:w_end] = roi

        new_instance_mask = torch.zeros((h_img, w_img), dtype=torch.bool)
        new_instance_mask[h_start:h_end, w_start:w_end] = torch.from_numpy(cut_mask)

        current_masks = base_target_dict["masks"]
        
        if current_masks.numel() > 0:
            occlusion_mask = new_instance_mask.unsqueeze(0).expand_as(current_masks)
            current_masks[occlusion_mask] = 0

        updated_masks = torch.cat([current_masks, new_instance_mask.unsqueeze(0)], dim=0)
        
        current_labels = base_target_dict["labels"]
        updated_labels = torch.cat([current_labels, torch.tensor([self.ood_label_id])], dim=0)

        current_iscrowd = base_target_dict["is_crowd"]
        updated_iscrowd = torch.cat([current_iscrowd, torch.tensor([False])], dim=0)

        base_target_dict["masks"] = tv_tensors.Mask(updated_masks)
        base_target_dict["labels"] = updated_labels
        base_target_dict["is_crowd"] = updated_iscrowd

        updated_img_tensor = tv_tensors.Image(torch.from_numpy(img_np).permute(2, 0, 1))

        return updated_img_tensor, base_target_dict

    def __getitem__(self, index: int):
        saved_transforms = self.transforms
        self.transforms = None

        try:
            img, target = super().__getitem__(index)
        finally:
            self.transforms = saved_transforms

        if (
            self.coco_loader 
            and len(self.coco_loader) > 0 
            and random.random() < self.ood_prob
        ):
            ood_img, ood_mask = self.coco_loader.get_random_sample()
            img, target = self.inject_ood(img, target, ood_img, ood_mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
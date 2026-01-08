import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random
import matplotlib.patches as mpatches

from eomt.datasets.cityscapes_coco_ood_semantic import CityscapesCocoOODDataset
from datasets.cityscapes_semantic import CityscapesSemantic

CITYSCAPES_ROOT = "../../Validation_Dataset/" 
COCO_ROOT = "../../Validation_Dataset/COCO"              
OOD_LABEL_ID = 254                  
OOD_PROB = 1.0    
NUM_EXAMPLES = 10


CLASS_NAMES = {
    0: 'Road', 1: 'Sidewalk', 2: 'Building', 3: 'Wall', 4: 'Fence',
    5: 'Pole', 6: 'Traffic Light', 7: 'Traffic Sign', 8: 'Vegetation',
    9: 'Terrain', 10: 'Sky', 11: 'Person', 12: 'Rider', 13: 'Car',
    14: 'Truck', 15: 'Bus', 16: 'Train', 17: 'Motorcycle', 18: 'Bicycle',
    254: 'OOD',
    255: 'Ignore'
}

COLORS = np.array([
    [128, 64, 128],  # 0: Road
    [244, 35, 232],  # 1: Sidewalk
    [70, 70, 70],    # 2: Building
    [102, 102, 156], # 3: Wall
    [190, 153, 153], # 4: Fence
    [153, 153, 153], # 5: Pole
    [250, 170, 30],  # 6: Traffic Light
    [220, 220, 0],   # 7: Traffic Sign
    [107, 142, 35],  # 8: Vegetation
    [152, 251, 152], # 9: Terrain
    [70, 130, 180],  # 10: Sky
    [220, 100, 60],   # 11: Person
    [255, 0, 0],     # 12: Rider
    [0, 0, 142],     # 13: Car
    [0, 0, 70],      # 14: Truck
    [0, 60, 100],    # 15: Bus
    [0, 80, 100],    # 16: Train
    [0, 0, 230],     # 17: Motorcycle
    [119, 11, 32],   # 18: Bicycle
])


COLOR_OOD = [255, 0, 0]    
COLOR_IGNORE = [0, 0, 0]

def decode_target_to_mask(target_dict, height, width):
    seg_map = np.full((height, width), 255, dtype=np.int32)
    
    masks = target_dict["masks"]
    labels = target_dict["labels"]

    for i in range(len(labels)):
        label_id = int(labels[i])
        
        mask = masks[i].numpy()
        
        if mask.max() > 0:
            seg_map[mask > 0] = label_id
            
    return seg_map

def colorize_mask(seg_map):
    h, w = seg_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(seg_map)
    
    for label in unique_labels:
        if label == OOD_LABEL_ID:
            c = COLOR_OOD
        elif label == 255:
            c = COLOR_IGNORE
        elif 0 <= label < 19:
            c = COLORS[label]
        else:
            c = [255, 255, 255] #Unknown White
            
        color_img[seg_map == label] = c
        
    return color_img, unique_labels

def main():
    target_parser = CityscapesSemantic.target_parser
    
    dataset = CityscapesCocoOODDataset(
        zip_path=Path(CITYSCAPES_ROOT, "leftImg8bit_trainvaltest.zip"),
        target_zip_path=Path(CITYSCAPES_ROOT, "gtFine_trainvaltest.zip"),
        img_folder_path_in_zip=Path("./leftImg8bit/train"),
        target_folder_path_in_zip=Path("./gtFine/train"),
        img_suffix=".png",
        target_suffix=".png",
        img_stem_suffix="leftImg8bit",
        target_stem_suffix="gtFine_labelIds",
        target_parser=target_parser,
        check_empty_targets=True,
        transforms=None,
        
        ood_enabled=True,
        ood_prob=1.0, # Force injection
        ood_coco_root=COCO_ROOT,
        ood_label_id=OOD_LABEL_ID
    )

    print(f"Dataset Loaded")

    count = 0
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if count >= NUM_EXAMPLES:
            break

        img_tensor, target_dict = dataset[idx]

        if OOD_LABEL_ID not in target_dict["labels"]:
            continue

        
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = img_np.astype(np.uint8)

        
        h, w = img_np.shape[:2]
        seg_map = decode_target_to_mask(target_dict, h, w)
        color_mask, unique_labels = colorize_mask(seg_map)

        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        
        axes[0].imshow(img_np)
        axes[0].set_title(f"Mixed Image (Index {idx})")
        axes[0].axis('off')

        
        axes[1].imshow(color_mask)
        axes[1].set_title("Generated Ground Truth")
        axes[1].axis('off')

        
        legend_patches = []
        for label in unique_labels:
            if label == 255: continue
            
            if label == OOD_LABEL_ID:
                color_norm = [c/255.0 for c in COLOR_OOD]
                label_text = f"ID {label}: {CLASS_NAMES.get(label, 'OOD')}"
            elif 0 <= label < 19:
                color_norm = [c/255.0 for c in COLORS[label]]
                label_text = f"ID {label}: {CLASS_NAMES.get(label, 'Unknown')}"
            else:
                continue

            patch = mpatches.Patch(color=color_norm, label=label_text)
            legend_patches.append(patch)

        axes[1].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        plt.tight_layout()
        plt.show()
        
        count += 1

if __name__ == "__main__":
    main()
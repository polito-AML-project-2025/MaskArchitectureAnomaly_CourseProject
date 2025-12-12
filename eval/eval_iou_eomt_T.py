# Code to calculate IoU (mean and per-class) for EoMT on Cityscapes with Temperature Scaling
# Adapted from eval_iou_eomt.py

import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import time
import yaml
import importlib
import warnings

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor
from torch.amp.autocast_mode import autocast

from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry
from PIL import Image

NUM_CLASSES = 19  # Cityscapes (EoMT uses 19, not 20)

input_transform_cityscapes = Compose([
    Resize(512, Image.BILINEAR),
    ToTensor(),
])
target_transform_cityscapes = Compose([
    Resize(512, Image.NEAREST),
    ToLabel(),
    Relabel(255, 19),   # ignore label to 19
])


def load_eomt_model(config_path, checkpoint_path, device='cuda'):
    """Load EoMT model following official approach"""
    
    # Add eomt to path
    eomt_path = os.path.join(os.path.dirname(__file__), '..', 'eomt')
    sys.path.insert(0, eomt_path)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Suppress Lightning warnings
    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )
    
    # Get image size from config
    img_size = config["data"]["init_args"]["img_size"]
    
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size, **encoder_cfg.get("init_args", {}))
    
    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,  # Disable for inference (from README)
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )
    
    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    model = lit_cls(
        img_size=img_size,
        num_classes=NUM_CLASSES,
        network=network,
        **model_kwargs,
    ).eval().to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    print("Model and weights LOADED successfully")
    
    return model, img_size


def infer_semantic_with_temperature(model, img, img_size, temperature, device='cuda'):
    """
    Semantic inference with temperature scaling
    Returns: predictions [H, W]
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img.to(device)]
        img_sizes = [img.shape[-2:]]
        
        # Official windowing approach
        crops, origins = model.window_imgs_semantic(imgs)
        
        # Forward pass
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear"
        )
        
        # Convert to per-pixel logits
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        
        # Revert windowing
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
        # Apply temperature scaling on final logits
        scaled_logits = logits[0] / temperature
        preds = scaled_logits.argmax(0).cpu()
    
    return preds


def main(args):
    device = 'cpu' if args.cpu else 'cuda'
    
    print(f"Loading EoMT model with Temperature = {args.temperature}...")
    model, img_size = load_eomt_model(args.config, args.checkpoint, device)
    
    print("Model loaded successfully")

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")
        return

    loader = DataLoader(
        cityscapes(args.datadir, input_transform_cityscapes, target_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=1,  # EoMT inference is per-image
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES)
    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        img = images[0]  # Get single image from batch
        label = labels[0]
        
        # Run EoMT inference with temperature
        preds = infer_semantic_with_temperature(model, img, img_size, args.temperature, device)
        
        # Add to IoU evaluator
        iouEvalVal.addBatch(preds.unsqueeze(0).unsqueeze(0), label.unsqueeze(0))
        
        filenameSave = filename[0].split("leftImg8bit/")[1] if "leftImg8bit/" in filename[0] else filename[0]
        print(step, filenameSave)

    # Calculate IoU
    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i] * 100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took", time.time() - start, "seconds")
    print(f"Temperature: {args.temperature}")
    print("=======================================")
    print("Per-Class IoU:")
    print(iou_classes_str[0], "Road")
    print(iou_classes_str[1], "sidewalk")
    print(iou_classes_str[2], "building")
    print(iou_classes_str[3], "wall")
    print(iou_classes_str[4], "fence")
    print(iou_classes_str[5], "pole")
    print(iou_classes_str[6], "traffic light")
    print(iou_classes_str[7], "traffic sign")
    print(iou_classes_str[8], "vegetation")
    print(iou_classes_str[9], "terrain")
    print(iou_classes_str[10], "sky")
    print(iou_classes_str[11], "person")
    print(iou_classes_str[12], "rider")
    print(iou_classes_str[13], "car")
    print(iou_classes_str[14], "truck")
    print(iou_classes_str[15], "bus")
    print(iou_classes_str[16], "train")
    print(iou_classes_str[17], "motorcycle")
    print(iou_classes_str[18], "bicycle")
    print("=======================================")
    iouStr = getColorEntry(iouVal) + '{:0.2f}'.format(iouVal * 100) + '\033[0m'
    print("MEAN IoU:", iouStr, "%")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--checkpoint', required=True, help='Path to EoMT checkpoint')
    parser.add_argument('--config', default='../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for scaling')
    parser.add_argument('--subset', default="val", help='val or train')
    parser.add_argument('--datadir', required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())

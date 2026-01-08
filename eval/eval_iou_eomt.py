# Code to calculate IoU (mean and per-class) for EoMT on Cityscapes
# Adapted from eval_iou.py - only model loading changed

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
import os
import sys
import time
import yaml
import importlib
import warnings

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image

from dataset import cityscapes
from transform import Relabel, ToLabel
from iouEval import iouEval, getColorEntry

NUM_CLASSES = 19  # Cityscapes (19 training classes)
NUM_CLASSES_EVAL = 20  # iouEval needs 20 (0-18 + 19 as ignore)


def main(args):
    device = 'cpu' if args.cpu else 'cuda'
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    print("Loading EoMT model...")
    
    # Add eomt to path
    eomt_path = os.path.join(os.path.dirname(__file__), '..', 'eomt')
    sys.path.insert(0, eomt_path)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    warnings.filterwarnings("ignore", message=r".*Attribute 'network' is an instance of.*")
    
    # Load checkpoint first to detect actual img_size
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Detect img_size from positional embeddings
    pos_embed_key = 'network.encoder.backbone.pos_embed'
    if pos_embed_key in state_dict:
        num_patches = state_dict[pos_embed_key].shape[1]
        # num_patches = (img_size / patch_size)^2, patch_size = 16
        img_size = int((num_patches ** 0.5) * 16)
        print(f"Detected img_size={img_size} from checkpoint")
    else:
        # Fallback
        img_size = 640
        print(f"Using default img_size={img_size}")
    
    # Create transforms with correct size (square images for EoMT)
    input_transform = Compose([
        Resize((img_size, img_size), Image.BILINEAR),  # Force square
        ToTensor(),
    ])
    target_transform = Compose([
        Resize((img_size, img_size), Image.NEAREST),  # Force square
        ToLabel(),
        Relabel(255, 19),
    ])
    
    # Convert img_size to tuple (height, width) as expected by official code
    img_size_tuple = (img_size, img_size)
    
    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=img_size_tuple, **encoder_cfg.get("init_args", {}))
    
    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )
    
    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    model = lit_cls(
        img_size=img_size_tuple,
        num_classes=NUM_CLASSES,
        network=network,
        **model_kwargs,
    )
    
    # Load checkpoint (already loaded earlier for img_size detection)
    model.load_state_dict(state_dict, strict=False)
    
    if not args.cpu:
        model = model.cuda()
    
    model.eval()
    print("Model and weights LOADED successfully")

    if not os.path.exists(args.datadir):
        print("Error: datadir could not be loaded")
        return

    loader = DataLoader(
        cityscapes(args.datadir, input_transform, target_transform, subset=args.subset),
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False
    )

    iouEvalVal = iouEval(NUM_CLASSES_EVAL)
    start = time.time()

    for step, (images, labels, filename, filenameGt) in enumerate(loader):
        if not args.cpu:
            labels = labels.cuda()

        # Convert normalized float tensor [0, 1] to uint8 [0, 255] for PIL
        imgs_uint8 = [(img * 255).to(torch.uint8) for img in images]
        
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            if not args.cpu:
                imgs_uint8 = [img.cuda() for img in imgs_uint8]
            
            img_sizes = [img.shape[-2:] for img in imgs_uint8]
            
            # Official semantic inference pipeline from inference.ipynb
            crops, origins = model.window_imgs_semantic(imgs_uint8)
            
            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], img_size_tuple, mode="bilinear"
            )
            
            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            
            # Stack predictions for the batch
            preds = torch.stack([logit.argmax(0) for logit in logits]).unsqueeze(1).cpu()

        iouEvalVal.addBatch(preds, labels.cpu())

        for i, fname in enumerate(filename):
            filenameSave = fname.split("leftImg8bit/")[1] if "leftImg8bit/" in fname else fname
            print(step * args.batch_size + i, filenameSave)

    iouVal, iou_classes = iouEvalVal.getIoU()

    iou_classes_str = []
    for i in range(iou_classes.size(0)):
        iouStr = getColorEntry(iou_classes[i]) + '{:0.2f}'.format(iou_classes[i] * 100) + '\033[0m'
        iou_classes_str.append(iouStr)

    print("---------------------------------------")
    print("Took", time.time() - start, "seconds")
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
    parser.add_argument('--config', required=True, help='Path to EoMT config yaml')
    parser.add_argument('--subset', default="val", help='val or train')
    parser.add_argument('--datadir', required=True, help='Path to Cityscapes dataset')
    parser.add_argument('--num-workers', type=int, default=2)  # Increased for faster data loading
    parser.add_argument('--batch-size', type=int, default=1)  # Keep at 1, windowing is sequential
    parser.add_argument('--cpu', action='store_true')

    main(parser.parse_args())

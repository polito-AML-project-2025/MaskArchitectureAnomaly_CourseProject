# EoMT Anomaly Evaluation Script
# Adapted from inference.ipynb + evalAnomaly.py structure

import os
import glob
import torch
import random
import yaml
import sys
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr
from sklearn.metrics import average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import importlib
import warnings
import json
from datetime import datetime
from peft import PeftModel

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose([
    Resize((512, 1024), Image.BILINEAR),
    ToTensor(),
])

target_transform = Compose([
    Resize((512, 1024), Image.NEAREST),
])


def load_eomt_model(config_path, checkpoint_path, lora_weights=None, device='cuda'):
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
    img_size = (1024,1024)#config["data"]["init_args"]["img_size"]
    num_classes = 19  # Cityscapes has 19 classes
    
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
        masked_attn_enabled=False,  # Disable for inference
        num_classes=num_classes,
        encoder=encoder,
        **network_kwargs,
    )
    
    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    
    model = lit_cls(
        img_size=img_size,
        num_classes=num_classes,
        network=network,
        **model_kwargs,
    ).eval().to(device)
    
    # Load checkpoint of model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    
    if lora_weights is not None:
        print('Loading lora weights')
        model.network = PeftModel.from_pretrained(model.network, lora_weights)
        model.network.eval().to(device)

    print("Model and weights LOADED successfully")
    
    return model, img_size


def infer_semantic_logits(model, img_tensor, img_size, device='cuda'):
    """
    Proper semantic inference
    Returns per-pixel logits [num_classes, H, W]
    """
    with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
        imgs = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]
        
        # Use official window_imgs_semantic method
        crops, origins = model.window_imgs_semantic(imgs)
        
        # Forward pass
        mask_logits_per_layer, class_logits_per_layer = model(crops)
        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], img_size, mode="bilinear"
        )
        
        # Convert to per-pixel logits (semantic inference)
        crop_logits = model.to_per_pixel_logits_semantic(
            mask_logits, class_logits_per_layer[-1]
        )
        
        # Revert windowing
        logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
        
    return logits[0].cpu().numpy()  # [num_classes, H, W]


def compute_anomaly_score(logits_np, method):
    """
    Compute anomaly scores from logits
    Args:
        logits_np: [num_classes, H, W] numpy array
        method: 'MSP', 'MaxLogit', 'MaxEntropy', or 'RbA'
    Returns:
        anomaly_score: [H, W] numpy array (higher = more anomalous)
    """
    logits = torch.from_numpy(logits_np).unsqueeze(0)  # [1, num_classes, H, W]
    
    if method == "MSP":
        # Maximum Softmax Probability
        probs = torch.nn.functional.softmax(logits, dim=1)
        anomaly_score = 1 - np.max(probs.squeeze(0).cpu().numpy(), axis=0)
        
    elif method == "MaxLogit":
        # Maximum Logit
        anomaly_score = -np.max(logits.squeeze(0).cpu().numpy(), axis=0)
        
    elif method == "MaxEntropy":
        # Normalized Entropy (normalized by log(num_classes))
        probs = torch.nn.functional.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        entropy = torch.div(
            torch.sum(-probs * torch.log(probs + 1e-10), dim=1),
            torch.log(torch.tensor(float(num_classes)))
        )
        anomaly_score = entropy.squeeze(0).cpu().numpy()
        
    elif method == "RbA":
        # RbA: Resynthesizing by Analysis
        # Apply tanh to bound logits, then sum across all classes
        logits_bounded = torch.nn.functional.tanh(logits)
        anomaly_score = torch.sum(-logits_bounded, dim=1).squeeze(0).cpu().numpy()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return anomaly_score


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--config",
        default="../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
    )
    parser.add_argument(
        "--lora_weights",
        default=None,
        help="Path to Lora weights checkpoint"
    )
    parser.add_argument(
        '--method',
        default='RbA',
        choices=['RbA', 'MSP', 'MaxLogit', 'MaxEntropy'],
    )
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    device = 'cpu' if args.cpu else 'cuda'
    
    # Load model
    model, img_size = load_eomt_model(args.config, args.checkpoint, args.lora_weights, device)
    
    anomaly_score_list = []
    ood_gts_list = []
    
    # Prepare results file
    if not os.path.exists('results_eomt.txt'):
        open('results_eomt.txt', 'w').close()
    file = open('results_eomt.txt', 'a')
    
    print(f"\n{'='*60}")
    print(f"Starting EoMT evaluation with method: {args.method}")
    print(f"{'='*60}\n")
    
    # Process each image
    for path in glob.glob(os.path.expanduser(str(args.input))):
        print(f"Processing: {path}")
        
        # Load and preprocess image
        img_pil = Image.open(path).convert('RGB')
        img_tensor = input_transform(img_pil)
        
        # Run inference (proper EoMT semantic inference)
        logits = infer_semantic_logits(model, img_tensor, img_size, device)
        
        # Compute anomaly score
        anomaly_result = compute_anomaly_score(logits, args.method)
        
        # Load ground truth
        pathGT = path.replace("images", "labels_masks")
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "RoadAnomaly21" in pathGT:
            pathGT = pathGT.replace("jpg", "png")
        if "FS_LostFound_full" in pathGT:
            pathGT = pathGT.replace("png", "png")
        
        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)
        
        # Process GT based on dataset
        if "RoadAnomaly" in pathGT and "RoadAnomaly21" not in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "RoadAnomaly21" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "FS_LostFound_full" in pathGT or "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        
        if 1 not in np.unique(ood_gts):
            continue
        
        ood_gts_list.append(ood_gts)
        anomaly_score_list.append(anomaly_result)
        
        torch.cuda.empty_cache()
    
    # Compute metrics
    ood_gts = np.array(ood_gts_list)
    anomaly_scores = np.array(anomaly_score_list)
    
    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)
    
    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]
    
    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
    
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))
    
    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)
    
    print(f'\n{"="*60}')
    print(f'EoMT Method: {args.method}')
    print(f'AUPRC score: {prc_auc*100.0:.2f}%')
    print(f'FPR@TPR95: {fpr*100.0:.2f}%')
    print(f'{"="*60}\n')
    
    file.write(f'\nEoMT Method: {args.method}  |  AUPRC: {prc_auc*100.0:.2f}%  |  FPR@TPR95: {fpr*100.0:.2f}%')
    file.close()
    
    # Save to Drive if RESULTS_PATH is available
    results_path = os.environ.get('RESULTS_PATH')
    if results_path and os.path.exists(results_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dict = {
            'model': 'EOMT',
            'method': args.method,
            'AUPRC': float(prc_auc * 100.0),
            'FPR95': float(fpr * 100.0),
            'timestamp': timestamp
        }
        
        json_filename = f"EOMT_{args.method}_{timestamp}.json"
        json_path = os.path.join(results_path, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f'Results also saved to Drive: {json_path}')


if __name__ == '__main__':
    main()

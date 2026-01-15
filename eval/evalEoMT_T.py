# EoMT Temperature Scaling Evaluation
# Two-stage approach: 1) Save logits, 2) Apply different temperatures

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
import pickle
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
    img_size = config["data"]["init_args"]["img_size"]
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
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
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


def save_logits(args):
    """Stage 1: Run model once and save all logits"""
    
    device = 'cpu' if args.cpu else 'cuda'
    
    # Load model
    model, img_size = load_eomt_model(args.config, args.checkpoint, args.lora_weights, device)
    
    logits_data = []
    
    print(f"\n{'='*60}")
    print(f"Stage 1: Saving logits from EoMT model")
    print(f"{'='*60}\n")
    
    # Process each image
    for path in glob.glob(os.path.expanduser(str(args.input))):
        print(f"Processing: {path}")
        
        # Load and preprocess image
        img_pil = Image.open(path).convert('RGB')
        img_tensor = input_transform(img_pil)
        
        # Run inference
        logits = infer_semantic_logits(model, img_tensor, img_size, device)
        
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
        
        logits_data.append({
            'logits': logits,
            'gt': ood_gts,
            'path': path
        })
        
        torch.cuda.empty_cache()
    
    # Save logits
    os.makedirs('saved_logits', exist_ok=True)
    dataset_name = os.path.basename(os.path.dirname(args.input))
    save_path = f'saved_logits/eomt_{dataset_name}_logits.pkl'
    
    with open(save_path, 'wb') as f:
        pickle.dump(logits_data, f)
    
    print(f"\nSaved {len(logits_data)} logits to: {save_path}")
    print(f"{'='*60}\n")


def compute_msp_with_temperature(logits_np, temperature):
    """
    Compute MSP anomaly score with temperature scaling
    Apply temperature on final pixel logits
    """
    logits = torch.from_numpy(logits_np).unsqueeze(0)  # [1, num_classes, H, W]
    
    # Apply temperature scaling on final logits
    scaled_logits = logits / temperature
    probs = torch.nn.functional.softmax(scaled_logits, dim=1)
    
    # MSP: 1 - max probability
    anomaly_score = 1 - np.max(probs.squeeze(0).cpu().numpy(), axis=0)
    
    return anomaly_score


def evaluate_with_temperature(args):
    """Stage 2: Load saved logits and apply different temperatures"""
    
    # Load saved logits
    dataset_name = os.path.basename(os.path.dirname(args.input))
    load_path = f'saved_logits/eomt_{dataset_name}_logits.pkl'
    
    if not os.path.exists(load_path):
        print(f"Error: Logits file not found: {load_path}")
        print("Please run with --save_logits first!")
        return
    
    with open(load_path, 'rb') as f:
        logits_data = pickle.load(f)
    
    print(f"\n{'='*60}")
    print(f"Stage 2: Evaluating with different temperatures")
    print(f"Loaded {len(logits_data)} samples from: {load_path}")
    print(f"{'='*60}\n")
    
    # Test different temperatures
    temperatures = args.temperatures if args.temperatures else [0.5, 0.75, 1.0, 1.1, 1.5, 2.0]
    
    results = []
    
    for temp in temperatures:
        print(f"\nTesting temperature: {temp}")
        
        anomaly_score_list = []
        ood_gts_list = []
        
        for data in logits_data:
            logits = data['logits']
            ood_gts = data['gt']
            
            # Compute anomaly score with temperature
            anomaly_score = compute_msp_with_temperature(logits, temp)
            
            anomaly_score_list.append(anomaly_score)
            ood_gts_list.append(ood_gts)
        
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
        
        results.append({
            'temperature': temp,
            'AUPRC': prc_auc * 100.0,
            'FPR95': fpr * 100.0
        })
        
        print(f'  AUPRC: {prc_auc*100.0:.2f}%')
        print(f'  FPR@TPR95: {fpr*100.0:.2f}%')
    
    # Print summary table
    print(f"\n{'='*60}")
    print("TEMPERATURE SCALING RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Temperature':<15} {'AUPRC (%)':<15} {'FPR95 (%)':<15}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['temperature']:<15.2f} {r['AUPRC']:<15.2f} {r['FPR95']:<15.2f}")
    print(f"{'='*60}\n")
    
    # Save results
    if not os.path.exists('results_eomt_temp.txt'):
        open('results_eomt_temp.txt', 'w').close()
    
    with open('results_eomt_temp.txt', 'a') as f:
        f.write(f"\n\nDataset: {dataset_name}\n")
        f.write(f"{'Temperature':<15} {'AUPRC (%)':<15} {'FPR95 (%)':<15}\n")
        f.write(f"{'-'*60}\n")
        for r in results:
            f.write(f"{r['temperature']:<15.2f} {r['AUPRC']:<15.2f} {r['FPR95']:<15.2f}\n")
    
    # Save to Drive if available
    results_path = os.environ.get('RESULTS_PATH')
    if results_path and os.path.exists(results_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"EOMT_TempScaling_{dataset_name}_{timestamp}.json"
        json_path = os.path.join(results_path, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump({
                'model': 'EOMT',
                'method': 'MSP_TempScaling',
                'dataset': dataset_name,
                'results': results,
                'timestamp': timestamp
            }, f, indent=4)
        
        print(f'Results saved to Drive: {json_path}')


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Path to dataset images"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to EoMT checkpoint"
    )
    parser.add_argument(
        "--lora_weights",
        default=None,
        help="Path to Lora weights checkpoint"
    )
    parser.add_argument(
        "--config",
        default="../eomt/configs/dinov2/cityscapes/semantic/eomt_base_640.yaml",
        help="Path to EoMT config"
    )
    parser.add_argument(
        '--save_logits',
        action='store_true',
        help='Stage 1: Save logits (run model once)'
    )
    parser.add_argument(
        '--eval_temp',
        action='store_true',
        help='Stage 2: Evaluate with different temperatures (load saved logits)'
    )
    parser.add_argument(
        '--temperatures',
        type=float,
        nargs='+',
        help='List of temperatures to test (default: [0.5, 0.75, 1.0, 1.1, 1.5, 2.0])'
    )
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    if args.save_logits:
        save_logits(args)
    elif args.eval_temp:
        evaluate_with_temperature(args)
    else:
        print("Error: Please specify either --save_logits or --eval_temp")
        print("\nUsage:")
        print("  Stage 1 (save logits): python evalEoMT_T.py --input <path> --checkpoint <path> --save_logits")
        print("  Stage 2 (eval temps):  python evalEoMT_T.py --input <path> --checkpoint <path> --eval_temp")


if __name__ == '__main__':
    main()

# Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
from erfnet import ERFNet
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr, plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

NUM_CHANNELS = 3
NUM_CLASSES = 20
# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

input_transform = Compose(
    [
        Resize((512, 1024), Image.BILINEAR),
        ToTensor(),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
    ]
)


def numpy_range_from_string(input_str):
    """Convert string like '0.5,3.0,0.1' to numpy range [0.5, 0.6, 0.7, ..., 2.9]"""
    try:
        parts = input_str.split(',')
        
        if len(parts) != 3:
            raise ValueError("Format must be 'start,end,step'")
            
        start, end, step = map(float, parts)
        
        if step == 0:
            raise ValueError("Step cannot be zero")
            
        return np.arange(start, end, step).tolist()

    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")


def compute_anomaly_score(result, method, temp=1.0):
    """
    Compute anomaly score from logits.
    
    Args:
        result: Tensor [1, num_classes, H, W] - logits from model
        method: str - 'MSP', 'MaxLogit', or 'MaxEntropy'
        temp: float - temperature for MSP (only used if method='MSP')
    
    Returns:
        numpy array [H, W] - anomaly scores
    """
    if method == "MSP":
        # Maximum Softmax Probability with temperature scaling
        probs = torch.nn.functional.softmax(result / temp, dim=1)
        anomaly_result = 1.0 - np.max(probs.squeeze(0).cpu().numpy(), axis=0)
        
    elif method == "MaxLogit":
        # Maximum Logit
        result_np = result.squeeze(0).cpu().numpy()
        anomaly_result = -np.max(result_np, axis=0)
        
    elif method == "MaxEntropy":
        # Normalized Entropy
        probs = torch.nn.functional.softmax(result, dim=1)
        num_classes = probs.shape[1]
        entropy = torch.div(
            torch.sum(-probs * torch.log(probs + 1e-10), dim=1),
            torch.log(torch.tensor(float(num_classes)))
        )
        anomaly_result = entropy.squeeze(0).cpu().numpy()
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return anomaly_result


def compute_metrics(anomaly_score_list, ood_gts_list):
    """Compute AUPRC and FPR@TPR95 metrics."""
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
    
    return prc_auc, fpr


def save_results_to_drive(results_dict, results_path):
    """Save results to Google Drive in JSON format."""
    if results_path and os.path.exists(results_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        json_filename = f"ERFNET_{results_dict['method']}_{results_dict['dataset']}_{timestamp}.json"
        json_path = os.path.join(results_path, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f'\n✓ Results saved to Drive: {json_path}')
        return json_path
    return None


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/content/Validation_Dataset/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images or glob pattern",
    )  
    parser.add_argument('--method', default='MSP', choices=['MSP', 'MaxLogit', 'MaxEntropy'])
    parser.add_argument('--temp', type=float, default=None, 
                        help='Single temperature value for MSP (default: 1.0)')
    parser.add_argument('--temp-range', type=str, default=None,
                        help='Temperature range for grid search: "start,end,step" (e.g., "0.5,3.0,0.1")')
    parser.add_argument('--loadDir', default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")
    parser.add_argument('--datadir', default="/content/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    
    # Validate temperature arguments
    if args.method == 'MSP':
        if args.temp is not None and args.temp_range is not None:
            raise ValueError("Cannot specify both --temp and --temp-range")
        if args.temp_range is not None:
            temps = numpy_range_from_string(args.temp_range)
            print(f"Temperature sweep enabled: {len(temps)} values from {temps[0]:.2f} to {temps[-1]:.2f}")
        elif args.temp is not None:
            temps = [args.temp]
            print(f"Using single temperature: {args.temp}")
        else:
            temps = [1.0]  # Default
            print("Using default temperature: 1.0")
    else:
        if args.temp is not None or args.temp_range is not None:
            print(f"Warning: Temperature parameter ignored for {args.method} method")
        temps = [1.0]  # Dummy value, won't be used
    
    # Results file
    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    # Load model
    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print("Loading model: " + modelpath)
    print("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if not args.cpu:
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                if name.startswith("module."):
                    own_state[name.split("module.")[-1]].copy_(param)
                else:
                    print(name, " not loaded")
                    continue
            else:
                own_state[name].copy_(param)
        return model

    model = load_my_state_dict(model, torch.load(weightspath, map_location=lambda storage, loc: storage))
    print("Model and weights LOADED successfully\n")
    model.eval()
    
    # Store logits and ground truths (needed for temperature sweep)
    logits_list = []
    ood_gts_list = []
    
    print(f"Processing images from: {args.input[0]}\n")
    
    for path in tqdm(glob.glob(os.path.expanduser(str(args.input[0]))), desc="Loading images"):
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float()
        if not args.cpu:
            images = images.cuda()
            
        with torch.no_grad():
            result = model(images)
        
        # Load ground truth
        pathGT = path.replace("images", "labels_masks")                
        if "RoadObsticle21" in pathGT:
            pathGT = pathGT.replace("webp", "png")
        if "fs_static" in pathGT:
            pathGT = pathGT.replace("jpg", "png")                
        if "RoadAnomaly" in pathGT:
            pathGT = pathGT.replace("jpg", "png")

        mask = Image.open(pathGT)
        mask = target_transform(mask)
        ood_gts = np.array(mask)

        # Dataset-specific label conversions
        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts == 2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts == 0), 255, ood_gts)
            ood_gts = np.where((ood_gts == 1), 0, ood_gts)
            ood_gts = np.where((ood_gts > 1) & (ood_gts < 201), 1, ood_gts)
        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts == 14), 255, ood_gts)
            ood_gts = np.where((ood_gts < 20), 0, ood_gts)
            ood_gts = np.where((ood_gts == 255), 1, ood_gts)

        # Only keep images with anomalies
        if 1 not in np.unique(ood_gts):
            continue              
        else:
            ood_gts_list.append(ood_gts)
            logits_list.append(result.cpu())  # Store on CPU to save GPU memory
        
        del result, ood_gts, mask
        torch.cuda.empty_cache()

    print(f"\n✓ Loaded {len(logits_list)} images with anomalies\n")
    
    # Extract dataset name
    dataset_name = "unknown"
    input_path = str(args.input[0])
    if "RoadObsticle21" in input_path:
        dataset_name = "RoadObsticle21"
    elif "RoadAnomaly21" in input_path:
        dataset_name = "RoadAnomaly21"
    elif "LostFound" in input_path or "LostAndFound" in input_path:
        dataset_name = "LostAndFound"
    elif "RoadAnomaly" in input_path:
        dataset_name = "RoadAnomaly"
    elif "fs_static" in input_path:
        dataset_name = "fs_static"
    elif "Streethazard" in input_path:
        dataset_name = "Streethazard"
    
    results_path = os.environ.get('RESULTS_PATH')
    
    # Temperature sweep mode (only for MSP)
    if args.method == 'MSP' and len(temps) > 1:
        print(f"{'='*70}")
        print(f"TEMPERATURE SWEEP for {args.method} on {dataset_name}")
        print(f"{'='*70}\n")
        
        prc_aucs = []
        fprs = []
        
        for temp in tqdm(temps, desc="Testing temperatures"):
            # Compute anomaly scores for this temperature
            anomaly_score_list = []
            for logits in logits_list:
                anomaly_score = compute_anomaly_score(logits, args.method, temp=temp)
                anomaly_score_list.append(anomaly_score)
            
            # Compute metrics
            prc_auc, fpr = compute_metrics(anomaly_score_list, ood_gts_list)
            prc_aucs.append(prc_auc * 100)
            fprs.append(fpr * 100)
            
            # Log to file
            file.write(f"\n{args.method} temp={temp:.4f}  |  AUPRC: {prc_auc*100:.2f}%  |  FPR@TPR95: {fpr*100:.2f}%")
        
        # Find best results
        best_auprc_idx = np.argmax(prc_aucs)
        best_fpr_idx = np.argmin(fprs)
        
        print(f"\n{'='*70}")
        print(f"TEMPERATURE SWEEP RESULTS")
        print(f"{'='*70}")
        print(f"Best AUPRC: {prc_aucs[best_auprc_idx]:.2f}% at temperature {temps[best_auprc_idx]:.4f}")
        print(f"Best FPR@TPR95: {fprs[best_fpr_idx]:.2f}% at temperature {temps[best_fpr_idx]:.4f}")
        print(f"{'='*70}\n")
        
        file.write(f"\n\nBest AUPRC: {prc_aucs[best_auprc_idx]:.2f}% at temperature {temps[best_auprc_idx]:.4f}")
        file.write(f"\nBest FPR@TPR95: {fprs[best_fpr_idx]:.2f}% at temperature {temps[best_fpr_idx]:.4f}\n")
        
        # Save sweep results to Drive
        if results_path:
            sweep_results = {
                'model': 'ERFNET',
                'method': args.method,
                'dataset': dataset_name,
                'sweep_type': 'temperature',
                'temperatures': temps,
                'AUPRC_values': prc_aucs,
                'FPR95_values': fprs,
                'best_AUPRC': float(prc_aucs[best_auprc_idx]),
                'best_AUPRC_temp': float(temps[best_auprc_idx]),
                'best_FPR95': float(fprs[best_fpr_idx]),
                'best_FPR95_temp': float(temps[best_fpr_idx]),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            save_results_to_drive(sweep_results, results_path)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(temps, prc_aucs, 'b-o', linewidth=2, markersize=4)
        ax1.axvline(temps[best_auprc_idx], color='r', linestyle='--', alpha=0.7, label=f'Best: T={temps[best_auprc_idx]:.2f}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel("Temperature", fontsize=12)
        ax1.set_ylabel("AUPRC (%)", fontsize=12)
        ax1.set_title(f"AUPRC vs Temperature\n{dataset_name}", fontsize=13, fontweight='bold')
        ax1.legend()
        
        ax2.plot(temps, fprs, 'r-o', linewidth=2, markersize=4)
        ax2.axvline(temps[best_fpr_idx], color='b', linestyle='--', alpha=0.7, label=f'Best: T={temps[best_fpr_idx]:.2f}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlabel("Temperature", fontsize=12)
        ax2.set_ylabel("FPR@TPR95 (%)", fontsize=12)
        ax2.set_title(f"FPR@TPR95 vs Temperature\n{dataset_name}", fontsize=13, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    else:
        # Single evaluation mode
        temp = temps[0]
        
        print(f"{'='*70}")
        print(f"Evaluating {args.method} on {dataset_name}")
        if args.method == 'MSP':
            print(f"Temperature: {temp}")
        print(f"{'='*70}\n")
        
        # Compute anomaly scores
        anomaly_score_list = []
        for logits in tqdm(logits_list, desc="Computing anomaly scores"):
            anomaly_score = compute_anomaly_score(logits, args.method, temp=temp)
            anomaly_score_list.append(anomaly_score)
        
        # Compute metrics
        prc_auc, fpr = compute_metrics(anomaly_score_list, ood_gts_list)
        
        print(f"\n{'='*70}")
        print(f"RESULTS")
        print(f"{'='*70}")
        print(f"Method: {args.method}")
        if args.method == 'MSP':
            print(f"Temperature: {temp}")
        print(f"Dataset: {dataset_name}")
        print(f"AUPRC: {prc_auc*100:.2f}%")
        print(f"FPR@TPR95: {fpr*100:.2f}%")
        print(f"{'='*70}\n")
        
        # Log to file
        temp_str = f" (T={temp})" if args.method == 'MSP' else ""
        file.write(f"\n{args.method}{temp_str}  |  {dataset_name}  |  AUPRC: {prc_auc*100:.2f}%  |  FPR@TPR95: {fpr*100:.2f}%")
        
        # Save to Drive
        if results_path:
            results_dict = {
                'model': 'ERFNET',
                'method': args.method,
                'dataset': dataset_name,
                'AUPRC': float(prc_auc * 100.0),
                'FPR95': float(fpr * 100.0),
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            if args.method == 'MSP':
                results_dict['temperature'] = float(temp)
            
            save_results_to_drive(results_dict, results_path)
    
    file.close()
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()
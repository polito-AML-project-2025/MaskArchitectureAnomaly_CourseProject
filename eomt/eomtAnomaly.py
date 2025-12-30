import yaml
from lightning import seed_everything
import torch
from torch.nn import functional as F
from torch.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import warnings
import importlib

import os
import cv2
import glob
import torch
import random
from PIL import Image
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm

from peft import PeftModel



seed = 42

# general reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
seed_everything(seed, verbose=False)

# gpu training specific
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

H = 1024
W = 2048

input_transform = Compose(
    [
        Resize((H, W), Image.BILINEAR),
        ToTensor(),
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((H, W), Image.NEAREST),
    ]
)

def MSP(logits, temp = 1):
    probs = torch.nn.functional.softmax(logits / temp, dim = 1)
    anomaly_score = 1 - np.max(probs.squeeze(0).data.cpu().numpy(), axis=0)
    return anomaly_score

def max_logits_anomaly(logits):
    anomaly_score = -np.max(logits.squeeze(0).data.cpu().numpy(), axis=0)
    return anomaly_score

def max_entropy_anomaly(logits):
    probs = torch.nn.functional.softmax(logits, dim = 1)
    entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=1), torch.log(torch.tensor(float(probs.shape[1]))))
    anomaly_score = entropy.squeeze(0).data.cpu().numpy()
    return anomaly_score

def rba_anomaly(logits):
    logits = torch.nn.functional.tanh(logits)
    anomaly_score = torch.sum(-logits, dim=1).squeeze(0).data.cpu().numpy()
    return anomaly_score

def logits_to_anomalyscores(logits_list, method, *args, **kwargs):
    return [method(logits, *args, **kwargs) for logits in logits_list]

def computeMetrics(anomaly_score_list, ood_mask, ind_mask):
    anomaly_scores = np.array(anomaly_score_list)

    ood_out = anomaly_scores[ood_mask]
    ind_out = anomaly_scores[ind_mask]

    ood_label = np.ones(len(ood_out))
    ind_label = np.zeros(len(ind_out))
        
    val_out = np.concatenate((ind_out, ood_out))
    val_label = np.concatenate((ind_label, ood_label))

    prc_auc = average_precision_score(val_label, val_out)
    fpr = fpr_at_95_tpr(val_out, val_label)

    return (prc_auc, fpr)

def logResults(file, prc_auc, fpr, description, toPrint=True):
    file.write( "\n")
    file.write('    AUPRC score:' + str(prc_auc*100.0) + '   FPR@TPR95:' + str(fpr*100.0) +'   ' +description)
    if(toPrint):
        print(description)
        print(f'AUPRC score: {prc_auc*100.0}')
        print(f'FPR@TPR95: {fpr*100.0}')

def numpy_range_from_string(input_str):
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        default="/home/shyam/Mask2Former/unk-eval/RoadObsticle21/images/*.webp",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )  
    #parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default=None)
    #parser.add_argument('--loadModel', default="erfnet.py")
    #parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    #parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--anomalyScore', default="msp")
    parser.add_argument('--temps', default=None)
    parser.add_argument('--config', default="configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml")
    parser.add_argument('--lora_weights', default=None)
    args = parser.parse_args()
    logits_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')


    #print(torch.cuda.is_available())
    device = 0  # TODO: change to the GPU you want to use
    config_path = args.config  # TODO: change to the config file

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = type('',(object,),{"img_size": (1024, 1024), "num_classes": 19})()

    #load model
    print("Loading model...")

    warnings.filterwarnings(
        "ignore",
        message=r".*Attribute 'network' is an instance of `nn\.Module` and is already saved during checkpointing.*",
    )

    # Load encoder
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    encoder_module_name, encoder_class_name = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(encoder_module_name), encoder_class_name)
    encoder = encoder_cls(img_size=data.img_size, **encoder_cfg.get("init_args", {}))

    # Load network
    network_cfg = config["model"]["init_args"]["network"]
    network_module_name, network_class_name = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(network_module_name), network_class_name)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=data.num_classes,
        encoder=encoder,
        **network_kwargs,
    )


    # Load Lightning module
    lit_module_name, lit_class_name = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module_name), lit_class_name)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    if "stuff_classes" in config["data"].get("init_args", {}):
        model_kwargs["stuff_classes"] = config["data"]["init_args"]["stuff_classes"]

    model = (
        lit_cls(
            img_size=data.img_size,
            num_classes=data.num_classes,
            network=network,
            **model_kwargs,
        )
        .eval()
        .to(device)
    )

    #load pre trained weight
    print("Loading weights...")
    
    name = config.get("trainer", {}).get("logger", {}).get("init_args", {}).get("name")
    '''
    if name is None:
        warnings.warn("No logger name found in the config. Please specify a model name.")
    else:
        try:

            

            is_dinov3 = "dinov3" in name

            if is_dinov3:
                model_kwargs["ckpt_path"] = state_dict_path
                model_kwargs["delta_weights"] = True

            model = (
                lit_cls(
                    img_size=data.img_size,
                    num_classes=data.num_classes,
                    network=network,
                    **model_kwargs,
                )
                .eval()
                .to(device)
            )

            if not is_dinov3:
                state_dict = torch.load(
                    state_dict_path, map_location=f"cuda:{device}", weights_only=True
                )
                model.load_state_dict(state_dict, strict=False)
            
            print ("Model and weights LOADED successfully")

        except RepositoryNotFoundError:
            warnings.warn(
                f"Pre-trained model not found for `{name}`. Please load your own checkpoint."
            )
    '''

    if args.loadWeights is not None:
        print('loading weights from file')
        state_dict_path = args.loadWeights
    else:
        print('loading weights from huggingface')
        state_dict_path = hf_hub_download(
            repo_id=f"tue-mps/{name}",
            filename="pytorch_model.bin",
        )

    ckpt = model._load_ckpt(state_dict_path, True)
    model.load_state_dict(ckpt, strict=False)

    if args.lora_weights is not None:
        print('load lora weights')
        model.network = PeftModel.from_pretrained(model.network, args.lora_weights)
        model.network.eval()

    #segment
    IGNORE_INDEX = 255


    def infer_semantic(img):
        with torch.no_grad(), autocast(dtype=torch.float16, device_type="cuda"):
            imgs = [img.to(device)]
            img_sizes = [img.shape[-2:] for img in imgs]
            crops, origins = model.window_imgs_semantic(imgs)

            mask_logits_per_layer, class_logits_per_layer = model(crops)
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], data.img_size, mode="bilinear"
            )

            crop_logits = model.to_per_pixel_logits_semantic(
                mask_logits, class_logits_per_layer[-1]
            )
            logits = model.revert_window_logits_semantic(crop_logits, origins, img_sizes)
            #preds = logits[0].argmax(0).cpu()

        #pred_array = preds.numpy()
        return logits[0]


    def plot_anomaly_results(img, pred_array):

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].imshow(img.permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Image")
        axes[1].imshow(pred_array)
        axes[1].set_title("Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.show()

    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB')))
        images = (images*255).to(torch.uint8)
        #images = images.permute(0,3,1,2)

        result = infer_semantic(images).unsqueeze(0).cpu()
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

        if "RoadAnomaly" in pathGT:
            ood_gts = np.where((ood_gts==2), 1, ood_gts)
        if "LostAndFound" in pathGT:
            ood_gts = np.where((ood_gts==0), 255, ood_gts)
            ood_gts = np.where((ood_gts==1), 0, ood_gts)
            ood_gts = np.where((ood_gts>1)&(ood_gts<201), 1, ood_gts)

        if "Streethazard" in pathGT:
            ood_gts = np.where((ood_gts==14), 255, ood_gts)
            ood_gts = np.where((ood_gts<20), 0, ood_gts)
            ood_gts = np.where((ood_gts==255), 1, ood_gts)

        if 1 not in np.unique(ood_gts):
            continue              
        else:
             ood_gts_list.append(ood_gts)
             logits_list.append(result)
        del result, ood_gts, mask, images
        torch.cuda.empty_cache()

    ood_gts = np.array(ood_gts_list)

    ood_mask = (ood_gts == 1)
    ind_mask = (ood_gts == 0)

    if(args.anomalyScore == 'msp' and args.temps is not None):
        temps = numpy_range_from_string(args.temps)
        prc_aucs = []
        fprs = []
        for temp in tqdm(temps):
            anomaly_score_list = logits_to_anomalyscores(logits_list, MSP, temp=temp)
            prc_auc, fpr = computeMetrics(anomaly_score_list, ood_mask, ind_mask)
            prc_aucs.append(prc_auc*100)
            fprs.append(fpr*100)
            description = args.anomalyScore + ' temp: '+ f"{temp:.4f}"
            logResults(file, prc_auc, fpr, '(' + description+')', toPrint=False)

        srt_best_auprc = 'best AUPRC: ' + str(max(prc_aucs)) + ' with temperature: ' + f"{temps[np.argmax(prc_aucs)]:.4f}"
        srt_best_fpr = 'best FPR: ' + str(min(fprs)) + ' with temperature: ' + f"{temps[np.argmin(fprs)]:.4f}"
        print(srt_best_auprc)
        print(srt_best_fpr)
        file.write('\n    ' + srt_best_auprc)
        file.write('\n    ' + srt_best_fpr)

        plt.plot(temps, prc_aucs)
        plt.grid(True)
        plt.xlabel("Temperature")
        plt.ylabel("AUPRC")
        plt.show()
        plt.plot(temps, fprs)
        plt.grid(True)
        plt.xlabel("Temperature")
        plt.ylabel("FPR")
        plt.show()
    else:
        if(args.anomalyScore == 'msp'):
            description = 'msp temp: 1'
            anomaly_score_list = logits_to_anomalyscores(logits_list, MSP, temp=1)
        elif(args.anomalyScore == 'ml'):
            description = 'ml'
            anomaly_score_list = logits_to_anomalyscores(logits_list, max_logits_anomaly)
        elif(args.anomalyScore == 'me'):
            description = 'me'
            anomaly_score_list = logits_to_anomalyscores(logits_list, max_entropy_anomaly)
        elif(args.anomalyScore == 'rba'):
            description = 'rba'
            anomaly_score_list = logits_to_anomalyscores(logits_list, rba_anomaly)
            for i in range(5):
                plt.imshow(anomaly_score_list[i])
                plt.show()
        else:
            raise ValueError("Error: unknown --anomalyScore value")
        prc_auc, fpr = computeMetrics(anomaly_score_list, ood_mask, ind_mask)
        logResults(file, prc_auc, fpr, '(' + description +')')

    file.write('\n')
    file.close()

if __name__ == '__main__':
    main()
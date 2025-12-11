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
from ood_metrics import fpr_at_95_tpr, calc_metrics, plot_roc, plot_pr,plot_barcode
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
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
        # Normalize([.485, .456, .406], [.229, .224, .225]),
    ]
)

target_transform = Compose(
    [
        Resize((512, 1024), Image.NEAREST),
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
    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="erfnet.py")
    parser.add_argument('--subset', default="val")  #can be val or train (must have labels)
    parser.add_argument('--datadir', default="/home/shyam/ViT-Adapter/segmentation/data/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--anomalyScore', default="msp")
    parser.add_argument('--temps', default=None)
    args = parser.parse_args()
    logits_list = []
    ood_gts_list = []

    if not os.path.exists('results.txt'):
        open('results.txt', 'w').close()
    file = open('results.txt', 'a')

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = ERFNet(NUM_CLASSES)

    if (not args.cpu):
        model = torch.nn.DataParallel(model).cuda()

    def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict elements
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
    print ("Model and weights LOADED successfully")
    model.eval()
    
    for path in glob.glob(os.path.expanduser(str(args.input[0]))):
        print(path)
        images = input_transform((Image.open(path).convert('RGB'))).unsqueeze(0).float().cuda()
        #images = images.permute(0,3,1,2)
        with torch.no_grad():
            result = model(images).cpu()

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
        del result, ood_gts, mask
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
        else:
            raise ValueError("Error: unknown --anomalyScore value")
        prc_auc, fpr = computeMetrics(anomaly_score_list, ood_mask, ind_mask)
        logResults(file, prc_auc, fpr, '(' + description +')')

    file.write('\n')
    file.close()

if __name__ == '__main__':
    main()

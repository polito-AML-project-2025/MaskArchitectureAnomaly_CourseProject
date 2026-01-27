import numpy as np
import os
from pathlib import Path
import subprocess
import pickle
from scipy.spatial.distance import cdist

def save_list_pickle(filename, new_elements):
    """Appends new elements to a pickle list."""
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            try:
                existing_data = pickle.load(f)
            except EOFError:
                existing_data = []
    else:
        existing_data = []

    existing_data.extend(new_elements)

    with open(filename, 'wb') as f:
        pickle.dump(existing_data, f)

def get_newest_folder_name(parent_folder_path):
    """Finds the most recently created directory."""
    directory = Path(parent_folder_path)
    if not directory.exists() or not directory.is_dir():
        return None
    subfolders = [f for f in directory.iterdir() if f.is_dir()]
    if not subfolders:
        return None
    return max(subfolders, key=os.path.getmtime).name

def get_next_hyperparameters(history, n_candidates=50):
    
    p_min, p_max = 1, 6
    a_min, a_max = 1.0, 10.0
    
    if not history:
        return np.random.uniform(p_min, p_max), np.random.uniform(a_min, a_max)
    
    existing_points = np.array([[row[0], row[1]] for row in history])
    
    bounds = np.array([[p_min, a_min], [p_max, a_max]])
    normalized_existing = (existing_points - bounds[0]) / (bounds[1] - bounds[0])
    
    candidates_norm = np.random.uniform(0, 1, (n_candidates, 2))
    
    dists = cdist(candidates_norm, normalized_existing, metric='euclidean')
    
    min_dists = dists.min(axis=1)
    
    best_candidate_idx = np.argmax(min_dists)
    best_candidate_norm = candidates_norm[best_candidate_idx]

    center_point = best_candidate_norm * (bounds[1] - bounds[0]) + bounds[0]
    
    p_sigma = (p_max - p_min) * 0.05
    a_sigma = (a_max - a_min) * 0.05
    
    next_p = np.random.normal(center_point[0], p_sigma)
    next_a = np.random.normal(center_point[1], a_sigma)

    next_p = np.clip(next_p, p_min, p_max)
    next_a = np.clip(next_a, a_min, a_max)

    return next_p, next_a


os.chdir("../")

lora_path = os.path.join(".", "lora_weights")
par_to_weights_path = os.path.join(lora_path, "par_to_weights.pkl")


if os.path.exists(par_to_weights_path):
    with open(par_to_weights_path, "rb") as f:
        try:
            history = pickle.load(f)
        except EOFError:
            history = []
else:
    history = []

print(f"Loaded {len(history)} existing runs.")


n_new_points = 49

for i in range(n_new_points):
    
    
    k, alpha = get_next_hyperparameters(history)
    
    print(f"\n--- Iteration {i+1}/{n_new_points} ---")
    print(f"Chosen Params: k={k:.4f}, alpha={alpha:.4f}")


    command = [
        "python", 
        "main.py", "fit",
        "-c", r".\configs\dinov2\cityscapes\semantic\eomt_base_640_cocomix.yaml",
        "--trainer.devices", "1",
        "--trainer.max_epochs", "1",
        "--trainer.limit_val_batches", "0",
        "--data.batch_size", "1",
        "--data.path", r"..\..\Validation_Dataset",
        "--data.init_args.ood_coco_root", r"..\..\Validation_Dataset\COCO", 
        "--model.init_args.lora_enabled", "True",
        "--model.init_args.modules_to_train", "['class_head', 'mask_head']",
        "--model.ckpt_path", r"..\..\epoch_106-step_19902_eomt.ckpt",
        "--model.load_ckpt_class_head", "True",
        "--model.network.masked_attn_enabled", "False",
        "--data.init_args.ood_prob", "1",
        "--model.init_args.rba_aplha", str(alpha),
        "--model.init_args.rba_coefficient", str(pow(10,-k)),
        "--trainer.logger", "False"
    ]
    
    subprocess.run(command)

    last_folder = get_newest_folder_name(lora_path)
    new_entry = (k, alpha, last_folder, None)
    save_list_pickle(par_to_weights_path, [new_entry])
    
    history.append(new_entry)

print("\nFinished all iterations.")
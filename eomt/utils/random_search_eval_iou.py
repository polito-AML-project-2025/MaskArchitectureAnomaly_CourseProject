import pickle
import os
import subprocess
import glob
import time

os.chdir("../")

lora_path = ".\lora_weights"
par_to_weights_path = lora_path + "\par_to_weights.pkl"
HISTORY_FILE = par_to_weights_path

EXTERNAL_SCRIPT_OUTPUT_DIR = "./eval_res_dic/" 
LORA_BASE_DIR = "./lora_weights/"
LORA_SUB_DIR = "epoch-00" 

IOU_SCRIPT_PATH = "main.py"
IOU_DATA_ROOT = r"..\..\Validation_Dataset"

CONFIG_PATH = "./configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
LOAD_WEIGHTS = "../../epoch_106-step_19902_eomt.ckpt"

DATASETS_TO_TEST = {
    "Cityscapes": r"..\..\Validation_Dataset",
}

SCORES_TO_CHECK = ["rba", "iou"]


def get_latest_file(directory, extension="*.pkl"):
    files = glob.glob(os.path.join(directory, extension))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def load_pickle(path):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"-> Master history saved to {path}")

def run_iou_task(lora_path, dataset_name, dataset_path):
    print(f"  [>>] {dataset_name}: Running {IOU_SCRIPT_PATH} for IoU...")

    cmd = [
        "python", IOU_SCRIPT_PATH, "validate",
        "-c", CONFIG_PATH,
        "--trainer.devices", "1",
        "--data.batch_size", "1",
        "--data.path", dataset_path,
        "--model.init_args.lora_enabled", "True",
        "--model.init_args.lora_weights_path", lora_path,
        "--model.ckpt_path", LOAD_WEIGHTS,
        "--model.load_ckpt_class_head", "True",
        "--model.network.masked_attn_enabled", "False",
        "--model.init_args.save_res_dict", "True"
    ]

    return execute_subprocess(cmd, dataset_name)

def execute_subprocess(cmd, dataset_name):
    try:
        start_time = time.time()
        # Run the command
        subprocess.run(cmd, check=True, shell=False)
        
        # Check for output
        latest_pkl = get_latest_file(EXTERNAL_SCRIPT_OUTPUT_DIR)
        
        if latest_pkl and os.path.getctime(latest_pkl) > start_time:
            print(f"  [V] Results merged from {os.path.basename(latest_pkl)}")
            return load_pickle(latest_pkl)
        else:
            print(f"  [X] Error: Could not find new result file for {dataset_name}")
            return {}

    except subprocess.CalledProcessError as e:
        print(f"  [X] Error running command: {e}")
        return {}
    except Exception as e:
        print(f"  [X] Unexpected error: {e}")
        return {}


def process_history():
    history = load_pickle(HISTORY_FILE)
    
    for i in range(len(history)):
        
        entry = list(history[i])
        p1, p2, lora_id, result_dict = entry
        
        if result_dict is None:
            result_dict = {}

        print(f"\nProcessing Entry {i+1}/{len(history)} | ID: {lora_id}")
        
        entry_modified = False
        full_lora_path = os.path.join(LORA_BASE_DIR, lora_id, LORA_SUB_DIR) + os.sep
        
        for dataset_name, dataset_path in DATASETS_TO_TEST.items():
            
            if dataset_name not in result_dict:
                result_dict[dataset_name] = {}

            existing_scores = result_dict[dataset_name].keys()
            
            missing_scores = [s for s in SCORES_TO_CHECK if s not in existing_scores]
            
            if not missing_scores:
                print(f"  [OK] {dataset_name}: All scores present.")
                continue

            iou_needed = 'iou' in missing_scores
            
            if iou_needed:
                new_results = run_iou_task(
                    lora_path=full_lora_path,
                    dataset_name=dataset_name,
                    dataset_path=dataset_path
                )
                if new_results:
                    result_dict[dataset_name].update(new_results)
                    entry_modified = True

        if entry_modified:
            entry[3] = result_dict
            history[i] = tuple(entry)
            save_pickle(history, HISTORY_FILE)

if __name__ == "__main__":
    process_history()
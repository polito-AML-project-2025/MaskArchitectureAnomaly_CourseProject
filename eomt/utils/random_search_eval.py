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

SCRIPT_PATH = "eomtAnomaly.py"
CONFIG_PATH = "./configs/dinov2/cityscapes/semantic/eomt_base_640.yaml"
LOAD_WEIGHTS = "../../epoch_106-step_19902_eomt.ckpt"

DATASETS_TO_TEST = {
    #"fs_static": r"..\..\Validation_Dataset\fs_static\images\*.jpg",
    #"RoadAnomaly21": r"..\..\Validation_Dataset\RoadAnomaly21\images\*.png",
    #'RoadAnomaly': r'..\..\Validation_Dataset\RoadAnomaly\images\*.jpg',
    'RoadObstacle21': r'..\..\Validation_Dataset\RoadObsticle21\images\*.webp',
#    'FS_LostFound': r'..\..\Validation_Dataset\FS_LostFound_full\images\*.png',
}

SCORES_TO_CHECK = ["rba"]#["msp", "ml" , "me", "rba"]


def get_latest_file(directory, extension="*.pkl"):
    """Finds the most recently created file in a directory."""
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



def process_history():
    history = load_pickle(HISTORY_FILE)
    
    
    for i in range(len(history)):
        
        entry = list(history[i])
        
        p1, p2, lora_id, result_dict = entry
        
        
        if result_dict is None:
            result_dict = {}

        print(f"\nProcessing Entry {i+1}/{len(history)} | ID: {lora_id}")
        
        entry_modified = False

        lora_path = os.path.join(LORA_BASE_DIR, lora_id, LORA_SUB_DIR)
        
        for dataset_name, dataset_path in DATASETS_TO_TEST.items():
            
            if dataset_name not in result_dict:
                result_dict[dataset_name] = {}

            existing_scores = result_dict[dataset_name].keys()
            
            missing_scores = [s for s in SCORES_TO_CHECK if s not in existing_scores]
            
            if not missing_scores:
                print(f"  [OK] {dataset_name}: All scores present.")
                continue


            scores_arg = ",".join(missing_scores)
            print(f"  [>>] {dataset_name}: Missing {missing_scores}. Running script...")

            cmd = [
                "python", SCRIPT_PATH,
                "--input", dataset_path,
                "--anomalyScore", scores_arg,
                "--config", CONFIG_PATH,
                "--loadWeights", LOAD_WEIGHTS,
                "--lora_weights", lora_path,
                "--save_res_dict", "True"
            ]

            try:
                start_time = time.time()
                subprocess.run(cmd, check=True, shell=False)
                latest_pkl = get_latest_file(EXTERNAL_SCRIPT_OUTPUT_DIR)
                
                if latest_pkl and os.path.getctime(latest_pkl) > start_time:
                    new_results = load_pickle(latest_pkl)
                    result_dict[dataset_name].update(new_results)
                    
                    entry_modified = True
                    print(f"  [V] Results merged from {os.path.basename(latest_pkl)}")
                else:
                    print(f"  [X] Error: Could not find new result file for {dataset_name}")

            except subprocess.CalledProcessError as e:
                print(f"  [X] Error running script: {e}")
            except Exception as e:
                print(f"  [X] Unexpected error: {e}")

        if entry_modified:
            entry[3] = result_dict
            history[i] = tuple(entry)
            save_pickle(history, HISTORY_FILE)

if __name__ == "__main__":
    process_history()
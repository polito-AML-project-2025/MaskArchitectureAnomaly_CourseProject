import pickle
import matplotlib.pyplot as plt
import argparse
import os
import re

def load_pickle(path):
    if not os.path.exists(path):
        print(f"Error: File {path} not found.")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def extract_epoch_number(epoch_str):
    """Extracts integer from strings like 'epoch-00', 'epoch-5'."""
    match = re.search(r'\d+', epoch_str)
    return int(match.group()) if match else -1

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary.
    Example: {'rba': {'auroc': 0.5}} -> {'rba_auroc': 0.5}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def plot_results(pkl_path, save_dir=None):
    data = load_pickle(pkl_path)
    if not data:
        return

    # 1. Sort epochs numerically
    sorted_epoch_keys = sorted(data.keys(), key=extract_epoch_number)
    
    if not sorted_epoch_keys:
        print("No epoch data found in file.")
        return

    # Create X-axis (Epoch numbers)
    epochs_x = [extract_epoch_number(k) for k in sorted_epoch_keys]

    # 2. Identify all unique datasets
    all_datasets = set()
    for e_key in data:
        all_datasets.update(data[e_key].keys())

    print(f"Found datasets: {all_datasets}")

    # 3. Create a plot for each dataset
    for dataset in all_datasets:
        print(f"Plotting results for: {dataset}")
        
        # Pre-process: Flatten metrics for this dataset across all epochs
        # This handles cases where 'rba' might be a dict {auroc: 0.9} in one epoch
        # and just 0.9 in another (though inconsistent data is rare).
        processed_data = {} # Key: epoch, Value: flattened_dict
        all_flat_metrics = set()

        for e_key in sorted_epoch_keys:
            if dataset in data[e_key]:
                raw_metrics = data[e_key][dataset]
                
                # Check if it's a dict (expected) or raw scalar
                if isinstance(raw_metrics, dict):
                    flat = flatten_dict(raw_metrics)
                else:
                    flat = {"Score": raw_metrics}
                
                processed_data[e_key] = flat
                all_flat_metrics.update(flat.keys())
            else:
                processed_data[e_key] = {}

        # Setup Plot
        plt.figure(figsize=(10, 6))
        
        # Plot a line for each FLATTENED metric
        for metric in sorted(list(all_flat_metrics)):
            metric_values = []
            
            for e_key in sorted_epoch_keys:
                # Get value or None if missing
                val = processed_data[e_key].get(metric, None)
                
                # Double check to ensure we are not plotting a dict
                # (This can happen if the flattening logic missed something specific, strictly safety)
                if isinstance(val, dict):
                    print(f"  [Warning] Skipping nested dict for {metric} at {e_key}")
                    val = None
                    
                metric_values.append(val)

            # Check if we have valid data to plot (at least one non-None number)
            if any(v is not None for v in metric_values):
                plt.plot(epochs_x, metric_values, marker='o', linestyle='-', label=metric)

        plt.title(f"Metric Evolution - {dataset}")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Save or Show
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{dataset}_results.png")
            plt.savefig(save_path)
            print(f"  -> Saved plot to {save_path}")
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize LoRA evaluation results.")
    parser.add_argument("--file", type=str, required=True, help="Path to evaluation_results.pkl")
    parser.add_argument("--output", type=str, default=None, help="Directory to save plots. If not set, displays plots.")
    
    args = parser.parse_args()

    plot_results(args.file, args.output)
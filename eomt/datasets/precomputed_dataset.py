import torch
from torch.utils.data import Dataset
import os
import glob
import re

class PrecomputedFeatureDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        
        self.file_paths = glob.glob(os.path.join(input_dir, "batch_*.pt"))
        if not self.file_paths:
            raise RuntimeError(f"No batch_*.pt files found in {input_dir}")


        self.file_paths.sort(key=lambda x: int(re.search(r'batch_(\d+).pt', x).group(1)))

        self.file_indices = []
        self.global_length = 0
        
        print("Indexing precomputed features...")
        for f_path in self.file_paths:
            data = torch.load(f_path, map_location="cpu", weights_only=True)
            n_samples = data["features"].shape[0]
            
            self.file_indices.append({
                "start": self.global_length,
                "end": self.global_length + n_samples,
                "path": f_path
            })
            self.global_length += n_samples
            
        print(f"Found {self.global_length} samples across {len(self.file_paths)} files.")

        self.cached_data = None
        self.cached_path = None

    def __len__(self):
        return self.global_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.global_length:
            raise IndexError("Index out of bounds")

        target_file_info = None
        for info in self.file_indices:
            if info["start"] <= idx < info["end"]:
                target_file_info = info
                break
        
        if target_file_info is None:
            raise RuntimeError(f"Could not locate index {idx}")

        if self.cached_path != target_file_info["path"]:
            self.cached_data = torch.load(target_file_info["path"], map_location="cpu", weights_only=True)
            self.cached_path = target_file_info["path"]

        local_idx = idx - target_file_info["start"]
        
        features = self.cached_data["features"][local_idx]
        rope = self.cached_data["rope"][local_idx]
        label = self.cached_data["labels"][local_idx]

        return (features, rope), label
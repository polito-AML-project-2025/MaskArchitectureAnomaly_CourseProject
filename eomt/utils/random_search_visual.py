import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.interpolate import griddata
import pickle
import os
import argparse

os.chdir("../")

parser = argparse.ArgumentParser()
parser.add_argument('--history_path', type=str, default=r".\lora_weights\par_to_weights.pkl")
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--dataset', type=str)
parser.add_argument('--score', type=str)
parser.add_argument('--metric', type=str)

args = parser.parse_args()

DATASET_NAME = args.dataset
SCORE_NAME = args.score
METRIC_NAME = args.metric

def load_pickle(path):
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return []
    with open(path, "rb") as f:
        return pickle.load(f)
    
HISTORY_FILE = args.history_path

res_dict = load_pickle(HISTORY_FILE)

results = []

for comb in res_dict:
    if DATASET_NAME not in comb[3]:
        continue

    if comb[3] is not None:
        results.append((comb[0], comb[1], comb[3][DATASET_NAME][SCORE_NAME][METRIC_NAME]))
    else:
        results.append((comb[0], comb[1], 0))

x = np.array([r[0] for r in results])
y = np.array([r[1] for r in results])
z = np.array([r[2] for r in results])

grid_x, grid_y = np.mgrid[min(x):max(x):50j, min(y):max(y):50j]

grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear') 

plt.figure(figsize=(10, 8))

img = plt.imshow(
    grid_z.T, 
    extent=(min(x), max(x), min(y), max(y)), 
    origin='lower', 
    aspect='auto',
    cmap='viridis' 
)

plt.scatter(x, y, c='black', s=20, marker='x', label='Real combination')

ax = plt.gca()

def log_tick_formatter(val, pos):
    return f"$10^{{-{int(val)}}}$"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

cbar = plt.colorbar(img)
cbar.set_label(args.score + " -> " + args.metric, fontsize=16)
cbar.ax.tick_params(labelsize=14)

plt.title(DATASET_NAME + ' (FPR@95)', fontsize=20, fontweight='bold')
plt.xlabel('RBA_loss_coeff', fontsize=16)
plt.ylabel('RBA_alpha', fontsize=16)

plt.legend(fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=14)

if args.save_path is not None:
    output_file = args.save_path + "/figure.png"
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

plt.show()
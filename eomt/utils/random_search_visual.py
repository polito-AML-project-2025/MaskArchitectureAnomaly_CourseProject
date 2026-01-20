import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import os
import argparse

os.chdir("../")


parser = argparse.ArgumentParser()
parser.add_argument('--history_path', type=str, default=r".\lora_weights\par_to_weights.pkl")
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
    if comb[3] is not None:
        results.append((comb[0], comb[1], comb[3][DATASET_NAME][SCORE_NAME][METRIC_NAME]))
    else:
        results.append((comb[0], comb[1], 0))



x = np.array([r[0] for r in results])
y = np.array([r[1] for r in results])
z = np.array([r[2] for r in results])

grid_x, grid_y = np.mgrid[min(x):max(x):500j, min(y):max(y):500j]


grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic') #linear


plt.figure(figsize=(10, 8))

img = plt.imshow(
    grid_z.T, 
    extent=(min(x), max(x), min(y), max(y)), 
    origin='lower', 
    aspect='auto',
    cmap='viridis' 
)


cbar = plt.colorbar(img)
cbar.set_label('Eval Metric')

plt.scatter(x, y, c='black', s=20, marker='x', label='Real combination')

plt.title('Random Search Results')
plt.xlabel('Par 1')
plt.ylabel('Par 2')
plt.legend()

plt.show()

'''
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(
    grid_x, grid_y, grid_z, 
    cmap='viridis', 
    edgecolor='none', 
    alpha=0.8,
    antialiased=True
)


ax.scatter(x, y, z, c='black', s=30, marker='x', label='Real combination', depthshade=False)


cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Eval Metric')


ax.set_title(f'Random Search Results ({METRIC_NAME})')
ax.set_xlabel('Par 1')
ax.set_ylabel('Par 2')
ax.set_zlabel(METRIC_NAME)


# ax.view_init(elev=30, azim=45) 

plt.legend()
plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

np.random.seed(42)


n_points = 50
par1_rand = np.random.uniform(0, 1, n_points)
par2_rand = np.random.uniform(0, 1, n_points)

scores = 10*np.sin(par1_rand * 3) + np.cos(par2_rand + 0.5) 

results = list(zip(par1_rand, par2_rand, scores))


x = np.array([r[0] for r in results])
y = np.array([r[1] for r in results])
z = np.array([r[2] for r in results])

grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]


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
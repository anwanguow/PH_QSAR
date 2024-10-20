import os
from ripser import Rips
from ripser import ripser
import numpy as np
import collections
import matplotlib.pyplot as plt
collections.Iterable = collections.abc.Iterable

r_left = 0
r_right = 5
num_points = 100

epsilon_range = (r_left, r_right)

def dist_mat(t):
    element = np.loadtxt(t, dtype=str, usecols=(0,), skiprows=2)
    x = np.loadtxt(t, dtype=float, usecols=(1), skiprows=2)
    y = np.loadtxt(t, dtype=float, usecols=(2), skiprows=2)
    z = np.loadtxt(t, dtype=float, usecols=(3), skiprows=2)
    Distance = np.zeros(shape=(len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            Distance[i][j] = np.sqrt(((x[i] - x[j])**2) + ((y[i] - y[j])**2) + ((z[i] - z[j])**2))
    return [Distance, element]

def compute_betti_numbers_at_radius(dgms, radius):
    betti_0 = np.sum((dgms[0][:, 0] <= radius) & (dgms[0][:, 1] > radius))
    betti_1 = np.sum((dgms[1][:, 0] <= radius) & (dgms[1][:, 1] > radius))
    betti_2 = np.sum((dgms[2][:, 0] <= radius) & (dgms[2][:, 1] > radius))
    return betti_0, betti_1, betti_2

all_betti_0 = []
all_betti_1 = []
all_betti_2 = []
radii = np.linspace(epsilon_range[0], epsilon_range[1], num_points)
directory = 'mols'

for filename in os.listdir(directory):
    if filename.endswith(".xyz"):
        file_path = os.path.join(directory, filename)
        D, elements = dist_mat(file_path)
        rips = Rips(maxdim=2)
        a = ripser(D, distance_matrix=True, maxdim=2)
        betti_0_list = []
        betti_1_list = []
        betti_2_list = []
        for radius in radii:
            betti_0, betti_1, betti_2 = compute_betti_numbers_at_radius(a['dgms'], radius)
            betti_0_list.append(betti_0)
            betti_1_list.append(betti_1)
            betti_2_list.append(betti_2)
        all_betti_0.append(betti_0_list)
        all_betti_1.append(betti_1_list)
        all_betti_2.append(betti_2_list)

plt.figure(figsize=(8, 5))
for i, betti_0_list in enumerate(all_betti_0):
    plt.plot(radii, betti_0_list)
plt.xlabel('Radius')
plt.ylabel('Betti-0 Number')
plt.title('Betti-0 Numbers as a Function of Radius')
plt.show()

plt.figure(figsize=(8, 5))
for i, betti_1_list in enumerate(all_betti_1):
    plt.plot(radii, betti_1_list, color=np.random.rand(3,))
plt.xlabel('Radius')
plt.ylabel('Betti-1 Number')
plt.title('Betti-1 Numbers as a Function of Radius')
plt.show()

plt.figure(figsize=(8, 5))
for i, betti_2_list in enumerate(all_betti_2):
    plt.plot(radii, betti_2_list, color=np.random.rand(3,))
plt.xlabel('Radius')
plt.ylabel('Betti-2 Number')
plt.title('Betti-2 Numbers as a Function of Radius')
plt.show()


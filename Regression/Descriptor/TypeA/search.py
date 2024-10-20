import pandas as pd
import numpy as np
import os
import networkx as nx

list_cc_path = "mols/list_cc"
mols_dir = "mols/mols"

try:
    molecule_info = pd.read_csv(list_cc_path, sep='\s+', header=None, usecols=[0, 1])
    molecule_info.columns = ['Name', 'Class']
except Exception as e:
    raise Exception(f"Error reading the list_cc file: {e}")
class_mapping = {'I': 0, 'II': 1, 'III': 2}
molecule_info['Class'] = molecule_info['Class'].map(class_mapping)

def read_xyz_to_graph(file_path, cutoff):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    num_atoms = int(lines[0].strip())
    atoms = []
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    adj_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(np.array(atoms[i][1:]) - np.array(atoms[j][1:]))
            if dist <= cutoff:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
    np.fill_diagonal(adj_matrix, 0)
    G = nx.from_numpy_array(adj_matrix)
    for i, atom in enumerate(atoms):
        G.nodes[i]['element'] = atom[0]
        G.nodes[i]['x'] = atom[1]
        G.nodes[i]['y'] = atom[2]
        G.nodes[i]['z'] = atom[3]
    return G, adj_matrix

def check_all_graphs_connected(cutoff):
    all_connected = True
    for _, row in molecule_info.iterrows():
        molecule_name = row['Name']
        xyz_path = os.path.join(mols_dir, f"{molecule_name}.xyz")
        if not os.path.exists(xyz_path):
            continue
        G, _ = read_xyz_to_graph(xyz_path, cutoff) 
        if not nx.is_connected(G):
            all_connected = False
            break
    return all_connected

def check_any_graph_fully_connected(cutoff):
    for _, row in molecule_info.iterrows():
        molecule_name = row['Name']
        xyz_path = os.path.join(mols_dir, f"{molecule_name}.xyz")
        if not os.path.exists(xyz_path):
            continue
        G, _ = read_xyz_to_graph(xyz_path, cutoff)
        if all([G.has_edge(i, j) for i in G.nodes for j in G.nodes if i != j]):
            return True
    return False

def find_minimum_r_values():
    r = 0
    step = 0.1
    min_r_connected = None
    first_fully_connected_r = None
    while min_r_connected is None or first_fully_connected_r is None:
        all_connected = check_all_graphs_connected(r)
        if min_r_connected is None and all_connected:
            min_r_connected = r
        if first_fully_connected_r is None and check_any_graph_fully_connected(r):
            first_fully_connected_r = r
        r += step  
    return min_r_connected, first_fully_connected_r

min_r_connected, first_fully_connected_r = find_minimum_r_values()

print(f"Minimum r for all graphs to be connected: {min_r_connected}")
print(f"First occurrence of a fully connected graph at r: {first_fully_connected_r}")

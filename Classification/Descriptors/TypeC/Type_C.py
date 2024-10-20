import pandas as pd
import numpy as np
import os
import networkx as nx
from sklearn.preprocessing import MinMaxScaler

a = 2.3
b = 4.1
n = 50

list_cc_path = "mols/list_cc"
mols_dir = "mols/mols"

molecule_info = pd.read_csv(list_cc_path, sep='\s+', header=None, usecols=[0, 1])
molecule_info.columns = ['Name', 'Class']
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
    return G

def degree_statistics(G):
    if G is None or len(G) == 0:
        return {
            'mean_degree': 0,
            'median_degree': 0,
            'std_degree': 0,
            'quantile_25': 0,
            'quantile_75': 0,
        }
    degrees = [d for _, d in G.degree()]
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    
    if max_degree == min_degree:
        return {
            'mean_degree': 0,
            'median_degree': 0,
            'std_degree': 0,
            'quantile_25': 0,
            'quantile_75': 0,
        }
    
    mean_degree = np.mean(degrees)
    median_degree = np.median(degrees)
    std_degree = np.std(degrees)
    quantile_25 = np.percentile(degrees, 25)
    quantile_75 = np.percentile(degrees, 75)

    mean_degree_norm = (mean_degree - min_degree) / (max_degree - min_degree)
    median_degree_norm = (median_degree - min_degree) / (max_degree - min_degree)
    std_degree_norm = (std_degree - min_degree) / (max_degree - min_degree)
    quantile_25_norm = (quantile_25 - min_degree) / (max_degree - min_degree)
    quantile_75_norm = (quantile_75 - min_degree) / (max_degree - min_degree)

    return {
        'mean_degree': mean_degree_norm,
        'median_degree': median_degree_norm,
        'std_degree': std_degree_norm,
        'quantile_25': quantile_25_norm,
        'quantile_75': quantile_75_norm,
    }

def calculate_all_degree_features(a, b, n):
    all_features = []
    r_values = np.linspace(a, b, n)
    for idx, row in molecule_info.iterrows():
        molecule_name = row['Name']
        molecule_class = row['Class']
        xyz_path = os.path.join(mols_dir, f"{molecule_name}.xyz")
        if not os.path.exists(xyz_path):
            continue
        molecule_features = []
        for r in r_values:
            G = read_xyz_to_graph(xyz_path, r)
            degree_stats = degree_statistics(G)
            feature_row_degree = list(degree_stats.values())
            molecule_features.extend(feature_row_degree)
        molecule_features.append(molecule_class)
        all_features.append(molecule_features)
    return all_features, r_values

all_features, r_values = calculate_all_degree_features(a, b, n)
degree_stat_columns = list(degree_statistics(None).keys())

feature_columns = []
for r in r_values:
    for col in degree_stat_columns:
        feature_columns.append(f"{col}_r{r:.2f}")

feature_columns.append("Class")
features_df = pd.DataFrame(all_features, columns=feature_columns)

scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features_df.iloc[:, :-1])
features_normalized_df = pd.DataFrame(features_normalized, columns=feature_columns[:-1])
features_normalized_df["Class"] = features_df["Class"]

output_path = "dataset/degree_dist_" + str(int(n)) + ".csv"
features_normalized_df.to_csv(output_path, index=False, header=False)

print(f"Done. Degree features with entropy saved to {output_path}.")

import os
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns

list_cc_path = "mols/list_cc"
mols_dir = "mols/mols"
molecule_info = pd.read_csv(list_cc_path, delim_whitespace=True)

def compute_persistent_homology_features(points, label):
    diagrams = ripser(points, maxdim=2)['dgms']
    stats = []
    for dim, dgm in enumerate(diagrams):
        if len(dgm) == 0:
            continue
        lengths = dgm[:, 1] - dgm[:, 0]
        lengths = lengths[np.isfinite(lengths)]
        stats.append({
            "dimension": dim,
            "num_barcodes": len(lengths),
            "max_length": np.max(lengths),
            "mean_length": np.mean(lengths),
            "persistent_entropy": entropy(lengths / np.sum(lengths)),
            "max_birth": np.max(dgm[:, 0]),
            "mean_death": np.mean(dgm[:, 1][np.isfinite(dgm[:, 1])]),
            "lifetime": lengths,
            "label": label
        })
    return stats

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()[2:]
    points = [list(map(float, line.split()[1:])) for line in lines]
    return np.array(points)

all_features = []
for idx, row in molecule_info.iterrows():
    molecule_name = row.iloc[0]
    label = row.iloc[1]
    file_path = os.path.join(mols_dir, f"{molecule_name}.xyz")
    points = read_xyz(file_path)
    if points is not None:
        features = compute_persistent_homology_features(points, label)
        all_features.extend(features)

features_df = pd.DataFrame(all_features)
description = features_df.groupby(['label', 'dimension']).describe()
description.to_csv("descriptive_statistics.csv")
order = ["I", "II", "III"]
dimension_map = {0: 'H0', 1: 'H1', 2: 'H2'}
color_map = {'H0': '#4B0082', 'H1': '#006400', 'H2': 'yellow'}

def plot_boxplot(df, column, y_label, filename):
    df['dimension_label'] = df['dimension'].map(dimension_map)
    plt.figure(figsize=(6, 6), dpi=300)
    sns.boxplot(x='label', y=column, hue='dimension_label', data=df, order=order, palette=color_map)
    plt.xlabel('Class', fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(title='', loc='best')
    plt.savefig(filename)
    plt.show()

plot_boxplot(features_df, 'persistent_entropy', 'Persistent Entropy', 'persistent_entropy.png')
plot_boxplot(features_df, 'mean_length', 'Mean Barcode Length', 'mean_barcode_length.png')
plot_boxplot(features_df, 'max_length', 'Maximum Barcode Length', 'max_barcode_length.png')
plot_boxplot(features_df, 'num_barcodes', 'Number of Barcodes', 'num_barcodes.png')
plot_boxplot(features_df, 'max_birth', 'Maximum Barcode Birth Time', 'max_birth_time.png')
plot_boxplot(features_df, 'mean_death', 'Mean Barcode Death Time', 'mean_death_time.png')
features_df_exploded = features_df.explode('lifetime')
plot_boxplot(features_df_exploded, 'lifetime', 'Barcode Lifetime', 'barcode_lifetime.png')


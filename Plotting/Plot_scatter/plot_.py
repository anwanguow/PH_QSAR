#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Type A": "data/reg_barcode_40.csv",
    "Type B": "data/reg_type_b_0.016.csv",
    "Type C": "data/reg_deg_dist_30.csv",
    "Type D": "data/reg_laplace_50.csv"
}

data = {name: pd.read_csv(filename) for name, filename in files.items()}

fig, axes = plt.subplots(2, 2, figsize=(12, 12), dpi=300)
axes = axes.flatten()

for i, (name, df) in enumerate(data.items()):
    ax = axes[i]
    ax.scatter(df['TrueValues'], df['PredictedValues'], alpha=0.5)
    ax.plot([df['TrueValues'].min(), df['TrueValues'].max()], 
            [df['TrueValues'].min(), df['TrueValues'].max()], 
            'r--', lw=2)
    ax.set_title(f"{name}", fontsize=22)
    ax.set_xlabel(r'True $T_g$', fontsize=20)
    ax.set_ylabel(r'Predicted $T_g$', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()
plt.savefig('scatter.png', dpi=300)
plt.show()

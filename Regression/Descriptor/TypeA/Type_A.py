import pandas as pd
import collections
from PH import CombinedFeature
collections.Iterable = collections.abc.Iterable
from sklearn.preprocessing import MinMaxScaler
import os

a = 1
b = 3
n = 50

def normalize_features(input_csv, output_csv):
    df = pd.read_csv(input_csv, header=None)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    
    scaler_features = MinMaxScaler()
    scaler_labels = MinMaxScaler()
    
    normalized_features = scaler_features.fit_transform(features)
    normalized_labels = scaler_labels.fit_transform(labels.values.reshape(-1, 1))
    
    normalized_df = pd.DataFrame(normalized_features)
    normalized_df['Target'] = normalized_labels
    
    normalized_df.to_csv(output_csv, index=False, header=False)

def build_dataset():
    with open('mols/list_reg', 'r') as file:
        lines = file.readlines()[1:]
    features = []
    targets = []
    
    for line in lines:
        parts = line.strip().split()
        filename_prefix = parts[0]
        target = float(parts[1])
        
        file_path = f'mols/mols/{filename_prefix}.xyz'
        feature_vector = CombinedFeature(file_path, a, b, n)
        features.append(feature_vector)
        targets.append(target)
    
    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame(targets, columns=['Target'])
    
    result_df = pd.concat([features_df, targets_df], axis=1)
    
    temp_csv = 'dataset/temp_combined_' + str(int(n)) + '.csv'
    result_df.to_csv(temp_csv, index=False, header=False)
    
    normalize_features(temp_csv, 'dataset/reg_barcode_' + str(int(n)) + '.csv')
    
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

build_dataset()

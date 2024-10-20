import os
import pandas as pd
from PH import PI_vector_h0h1h2
import collections
from sklearn.preprocessing import MinMaxScaler

collections.Iterable = collections.abc.Iterable

pixel_m = 40
pixel_n = pixel_m

sigma = 0.02
Max = 2.5
Min = -0.1

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
        feature_vector = PI_vector_h0h1h2(file_path, pixelx=pixel_m, pixely=pixel_n, myspread=sigma, myspecs={"maxBD": Max, "minBD": Min}, showplot=False)
        features.append(feature_vector)
        targets.append(target)
    
    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame(targets, columns=['Tg'])
    result_df = pd.concat([features_df, targets_df], axis=1)
    raw_data_path = 'dataset/data_raw_' + str(sigma) + '.csv'
    result_df.to_csv(raw_data_path, index=False, header=False)
    normalized_data_path = 'dataset/reg_type_b_' + str(sigma) + '.csv'
    normalize_features(raw_data_path, normalized_data_path)

    if os.path.exists(raw_data_path):
        os.remove(raw_data_path)

def normalize_features(input_csv, output_csv):
    df = pd.read_csv(input_csv, header=None)
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)
    labels = labels.values.reshape(-1, 1)
    normalized_labels = scaler.fit_transform(labels).flatten()
    normalized_df = pd.DataFrame(normalized_features)
    normalized_df['Tg'] = normalized_labels
    normalized_df.to_csv(output_csv, index=False, header=False)

build_dataset()

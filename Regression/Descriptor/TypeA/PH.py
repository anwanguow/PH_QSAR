from ripser import Rips
from ripser import ripser
rips = Rips(maxdim=2)
import numpy as np

def dist_mat(t):
    element=np.loadtxt(t,dtype=str,usecols=(0,), skiprows=2)
    x=np.loadtxt(t,dtype=float,usecols=(1), skiprows=2)
    y=np.loadtxt(t,dtype=float,usecols=(2),skiprows=2)
    z=np.loadtxt(t,dtype=float,usecols=(3),skiprows=2)
    Distance=np.zeros(shape=(len(x),len(x)))
    for i in range(0,len(x)):
        for j in range(0,len(x)):
            Distance[i][j]=np.sqrt(((x[i]-x[j])**2) + ((y[i]-y[j])**2) + ((z[i]-z[j]) **2))
    return [Distance, element]

def BettiNumberSeries(xyz, a, b, n):
    D, elements = dist_mat(xyz)
    diagrams = rips.fit_transform(D, distance_matrix=True)
    filtrations = np.linspace(a, b, n)
    betti_numbers = []

    for filtration in filtrations:
        betti_h0 = np.sum((diagrams[0][:, 1] > filtration) | np.isinf(diagrams[0][:, 1])) - np.sum(diagrams[0][:, 0] > filtration)
        betti_h1 = np.sum((diagrams[1][:, 1] > filtration) | np.isinf(diagrams[1][:, 1])) - np.sum(diagrams[1][:, 0] > filtration)
        betti_h2 = np.sum((diagrams[2][:, 1] > filtration) | np.isinf(diagrams[2][:, 1])) - np.sum(diagrams[2][:, 0] > filtration)
        betti_numbers.extend([betti_h0, betti_h1, betti_h2])  
    feature_vector = np.array(betti_numbers)
    return feature_vector

def Stat_quant(xyz):
    D, elements = dist_mat(xyz)
    finite_distances = D[np.isfinite(D)]
    max_radius = np.max(finite_distances)
    results = ripser(D, distance_matrix=True, maxdim=2, thresh=max_radius)
    diagrams = results['dgms']
    
    statistics = []

    for i in range(3):  # for H0, H1, H2 homology classes
        if i < len(diagrams):
            barcode_lengths = []
            for interval in diagrams[i]:
                birth, death = interval
                if np.isinf(death):
                    death = max_radius
                barcode_lengths.append(death - birth)
            
            if barcode_lengths:
                min_val = np.min(barcode_lengths)
                max_val = np.max(barcode_lengths)
                mean_val = np.mean(barcode_lengths)
                std_val = np.std(barcode_lengths)
                sum_val = np.sum(barcode_lengths)
            else:
                min_val = max_val = mean_val = std_val = sum_val = 0
            statistics.extend([min_val, max_val, mean_val, std_val, sum_val])
        else:
            statistics.extend([0, 0, 0, 0, 0])
    return np.array(statistics)


def CombinedFeature(xyz, a, b, n):
    betti_features = BettiNumberSeries(xyz, a, b, n)
    stat_features = Stat_quant(xyz)
    combined_feature_vector = np.concatenate((betti_features, stat_features))
    return combined_feature_vector





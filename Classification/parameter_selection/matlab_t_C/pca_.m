N = 50;

datafile = 'data/degree_dist_' + string(N) + '.csv';
load_pca_datafile = 'data_pca/degree_dist_' + string(N) + '_pca.csv';

rng(42);  

data = readtable(datafile);
X = data{:,1:end-1};
y = data{:,end};

rt = 1;

[coeff, score, ~, ~, explained] = pca(X);
numComponents = find(cumsum(explained) >= 100 * rt, 1);
X_pca = score(:, 1:numComponents);

if isempty(X_pca)
    rt = 0.9999;
    numComponents = find(cumsum(explained) >= 100 * rt, 1);
    X_pca = score(:, 1:numComponents);
end

data_pca = [X_pca, y];
data_pca_table = array2table(data_pca);
writetable(data_pca_table, load_pca_datafile);

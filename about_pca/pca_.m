datafile = 'data/test_data.csv';
data = readtable(datafile);
X = data{:,1:end-1};

rt = 1;
[coeff, score, ~, ~, explained] = pca(X);
numComponents = find(cumsum(explained) >= 100 * rt, 1);

if ~isempty(numComponents)
    X_pca = score(:, 1:numComponents)';
else
    rt = 0.9999;
    numComponents = find(cumsum(explained) >= 100 * rt, 1);
    X_pca = score(:, 1:numComponents)';
end
size(X_pca, 1)


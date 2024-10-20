N = 50;

datafile = 'data/reg_deg_dist_' + string(N) + '.csv';
load_pca_datafile = 'data_pca/reg_deg_dist_' + string(N) + '_pca.csv';

% 设置全局随机数种子
rng(42);  % 设置初始种子，影响整个过程的初始随机性

% 1. 读取数据并执行 PCA
data = readtable(datafile);
X = data{:,1:end-1};
y = data{:,end};

% 解释率阈值
rt = 1;

% 执行 PCA
[coeff, score, ~, ~, explained] = pca(X);
numComponents = find(cumsum(explained) >= 100 * rt, 1);
X_pca = score(:, 1:numComponents);

% 将降维后的特征矩阵和标签合并
data_pca = [X_pca, y];

% 将结果保存为 CSV 文件
data_pca_table = array2table(data_pca);
writetable(data_pca_table, load_pca_datafile);

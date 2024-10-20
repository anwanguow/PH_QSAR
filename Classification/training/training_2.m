% Type B

load_model = 'models/PI_40_pure_0.002.mat';
load_pca_datafile = 'data_pca/PI_40_pure_0.002_pca.csv';

load(load_model, 'net', 'dropoutRate');
data_pca = readtable(load_pca_datafile);
X_pca = data_pca{:,1:end-1}';
y = data_pca{:,end};
Y = full(ind2vec(y' + 1));

rng(2);

[trainConfMat, testConfMat] = looCrossValidateWithoutDropout(net, X_pca, Y);

testAccuracy = sum(diag(testConfMat)) / sum(testConfMat(:));
trainAccuracy = sum(diag(trainConfMat)) / sum(trainConfMat(:));
trainMCC = calculateMCC(trainConfMat);
testMCC = calculateMCC(testConfMat);

fprintf('Train Confusion Matrix:\n');
disp(trainConfMat);
fprintf('Train Accuracy: %.2f%%\n', trainAccuracy * 100);
fprintf('Train MCC: %.2f\n', trainMCC);
fprintf('Test Confusion Matrix:\n');
disp(testConfMat);
fprintf('Test Accuracy: %.2f%%\n', testAccuracy * 100);
fprintf('Test MCC: %.2f\n', testMCC);

function net = trainWithoutDropout(net, X, Y)
    for i = 1:numel(net.layers)-1
        if isfield(net.layers{i}, 'transferFcn')
            net.layers{i}.transferFcn = str2func(net.layers{i}.transferFcn);
        end
    end
    net.trainParam.showWindow = false;
    net = train(net, X, Y);
end

function [trainConfMat, testConfMat] = looCrossValidateWithoutDropout(net, X, Y)
    numSamples = size(X, 2);
    numClasses = size(Y, 1);
    trainConfMat = zeros(numClasses, numClasses);
    testConfMat = zeros(numClasses, numClasses);
    for i = 1:numSamples
        trainIdx = true(1, numSamples);
        trainIdx(i) = false;
        rng(100 + i);
        net = trainWithoutDropout(net, X(:,trainIdx), Y(:,trainIdx));
        trainPredictions = net(X(:,trainIdx));
        [~, trainPredClass] = max(trainPredictions, [], 1);
        [~, trainTrueClass] = max(Y(:,trainIdx), [], 1);
        for j = 1:length(trainTrueClass)
            trainConfMat(trainTrueClass(j), trainPredClass(j)) = trainConfMat(trainTrueClass(j), trainPredClass(j)) + 1;
        end

        testPredictions = net(X(:,~trainIdx));
        [~, testPredClass] = max(testPredictions, [], 1);
        [~, testTrueClass] = max(Y(:,~trainIdx), [], 1);
        testConfMat(testTrueClass, testPredClass) = testConfMat(testTrueClass, testPredClass) + 1;
    end
end

function mcc = calculateMCC(confMat)
    c = sum(confMat(:));
    s1 = sum(diag(confMat));
    row_sum = sum(confMat, 2);
    col_sum = sum(confMat, 1);
    s2 = sum(row_sum .* col_sum');
    s3 = sum(row_sum .^ 2);
    s4 = sum(col_sum .^ 2);
    numerator = c * s1 - s2;
    denominator = sqrt((c^2 - s3) * (c^2 - s4));
    
    if denominator == 0
        mcc = 0;
    else
        mcc = numerator / denominator;
    end
end

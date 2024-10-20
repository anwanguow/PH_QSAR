number_ = 0.016;

save_model = 'models/PI_40_pure_' + string(number_) + '.mat';
results_file = 'results/PI_40_pure_' + string(number_) + '.txt';
load_pca_datafile = 'data_pca/PI_40_pure_' + string(number_) + '_pca.csv';

rng(42);
data_pca = readtable(load_pca_datafile);
X_pca = data_pca{:,1:end-1}';
y = data_pca{:,end};
Y = full(ind2vec(y' + 1));

rng(1);
vars = [...
    optimizableVariable('hiddenLayerSize',[1,800],'Transform','log'), ...
    optimizableVariable('numHiddenLayers',[1,3],'Type','integer'), ...
    optimizableVariable('learningRate',[1e-4,1e-1],'Transform','log'), ...
    optimizableVariable('dropoutRate',[0,0.5],'Transform','none')];
steps_bo = 100;
results = bayesopt(@(params) looLoss(params, X_pca, Y), ...
    vars, 'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 0, 'MaxObjectiveEvaluations', steps_bo);

optimalParams = bestPoint(results);
hiddenLayerSize = round(optimalParams.hiddenLayerSize);
numHiddenLayers = optimalParams.numHiddenLayers;
learningRate = optimalParams.learningRate;
dropoutRate = optimalParams.dropoutRate;

rng(2);

hiddenLayers = repmat(hiddenLayerSize, 1, numHiddenLayers);
net = patternnet(hiddenLayers);
net.trainParam.lr = learningRate;
net.trainParam.showWindow = false;
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

save(save_model, 'net', 'dropoutRate');
fid = fopen(results_file, 'w');
fprintf(fid, 'Train Confusion Matrix:\n');
fprintfMatrix(fid, trainConfMat);
fprintf(fid, 'Train Accuracy: %.2f%%\n', trainAccuracy * 100);
fprintf(fid, 'Train MCC: %.2f\n', trainMCC);
fprintf(fid, 'Test Confusion Matrix:\n');
fprintfMatrix(fid, testConfMat);
fprintf(fid, 'Test Accuracy: %.2f%%\n', testAccuracy * 100);
fprintf(fid, 'Test MCC: %.2f\n', testMCC);
fclose(fid);

function loss = looLoss(params, X, Y)
    numSamples = size(X, 2);
    valLoss = zeros(1, numSamples);
    for i = 1:numSamples
        trainIdx = true(1, numSamples);
        trainIdx(i) = false;
        hiddenLayerSize = round(params.hiddenLayerSize);
        hiddenLayers = repmat(hiddenLayerSize, 1, params.numHiddenLayers);
        net = patternnet(hiddenLayers);
        net.trainParam.lr = params.learningRate;
        net.trainParam.showWindow = false;
        net = trainWithDropout(net, X(:,trainIdx), Y(:,trainIdx), params.dropoutRate);
        predictions = net(X(:,~trainIdx));
        trueClass = Y(:,~trainIdx);
        valLoss(i) = crossentropy(predictions, trueClass);
    end
    loss = mean(valLoss);
end

function net = trainWithDropout(net, X, Y, dropoutRate)
    for i = 1:numel(net.layers)-1
        if isfield(net.layers{i}, 'transferFcn')
            net.layers{i}.transferFcn = @(x) dropoutActivation(x, dropoutRate, net.layers{i}.transferFcn);
        end
    end
    net.trainParam.showWindow = false;
    net = train(net, X, Y);
end

function y = dropoutActivation(x, dropoutRate, transferFcn)
    dropoutMask = rand(size(x)) > dropoutRate;
    x = x .* dropoutMask;
    y = feval(transferFcn, x);
end

function loss = crossentropy(predictions, trueClass)
    epsilon = 1e-10;
    predictions = max(min(predictions, 1 - epsilon), epsilon);
    loss = -sum(trueClass .* log(predictions));
end

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

function fprintfMatrix(fid, mat)
    [rows, cols] = size(mat);
    for i = 1:rows
        for j = 1:cols
            fprintf(fid, '%d ', mat(i, j));
        end
        fprintf(fid, '\n');
    end
end

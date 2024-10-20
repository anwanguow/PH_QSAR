number_ = 0.001;

% Set file paths
save_model = 'models/reg_type_b_' + string(number_) + '.mat';
results_file = 'results/reg_type_b_' + string(number_) + '.txt';
load_pca_datafile = 'data_pca/reg_type_b_' + string(number_) + '_pca.csv';

% Set global random seed
rng(42);  % Set initial seed to control randomness throughout the process

% 1. Load PCA processed data
data_pca = readtable(load_pca_datafile);
X_pca = data_pca{:,1:end-1}';
y = data_pca{:,end}';

% 2. Bayesian Optimization - Set random seed
rng(1);  % Ensure randomness during Bayesian optimization
vars = [...
    optimizableVariable('hiddenLayerSize',[1,800],'Transform','log'), ...  % Reduce hidden layer size range
    optimizableVariable('numHiddenLayers',[1,3],'Type','integer'), ...  % Limit number of hidden layers
    optimizableVariable('learningRate',[1e-4,1e-1],'Transform','log'), ...
    optimizableVariable('dropoutRate',[0,0.5],'Transform','none')];
steps_bo = 20;  % Reduce Bayesian optimization steps
results = bayesopt(@(params) looLoss(params, X_pca, y), ...
    vars, 'AcquisitionFunctionName', 'expected-improvement-plus', 'Verbose', 0, 'MaxObjectiveEvaluations', steps_bo);

% 3. Obtain optimal hyperparameters
optimalParams = bestPoint(results);
hiddenLayerSize = round(optimalParams.hiddenLayerSize);
numHiddenLayers = optimalParams.numHiddenLayers;
learningRate = optimalParams.learningRate;
dropoutRate = optimalParams.dropoutRate;

% 4. Set new random seed for LOO loop
rng(2);  % Ensure different randomness for LOO loop compared to Bayesian optimization

% 5. Set up the neural network
hiddenLayers = repmat(hiddenLayerSize, 1, numHiddenLayers);
net = feedforwardnet(hiddenLayers);  % Use a neural network architecture for regression tasks
net.trainFcn = 'trainscg';  % Use a memory-efficient algorithm
net.trainParam.lr = learningRate;
net.trainParam.showWindow = false;

% 6. Perform LOO loop for final evaluation
[trainMSE, testMSE, trainPCC, testPCC] = looCrossValidateWithoutDropout(net, X_pca, y);

% 7. Compute and display final results
fprintf('Train MSE: %.4f\n', trainMSE);
fprintf('Train PCC: %.4f\n', trainPCC);
fprintf('Test MSE: %.4f\n', testMSE);
fprintf('Test PCC: %.4f\n', testPCC);

% 8. Save final model and results
save(save_model, 'net', 'dropoutRate');
fid = fopen(results_file, 'w');
fprintf(fid, 'Train MSE: %.4f\n', trainMSE);
fprintf(fid, 'Train PCC: %.4f\n', trainPCC);
fprintf(fid, 'Test MSE: %.4f\n', testMSE);
fprintf(fid, 'Test PCC: %.4f\n', testPCC);
fclose(fid);

% Function section

function loss = looLoss(params, X, y)
    numSamples = size(X, 2);
    valLoss = zeros(1, numSamples);
    for i = 1:numSamples
        trainIdx = true(1, numSamples);
        trainIdx(i) = false;
        hiddenLayerSize = round(params.hiddenLayerSize);
        hiddenLayers = repmat(hiddenLayerSize, 1, params.numHiddenLayers);
        net = feedforwardnet(hiddenLayers);
        net.trainFcn = 'trainscg';  % Use a memory-friendly algorithm
        net.trainParam.lr = params.learningRate;

        % Disable GUI
        net.trainParam.showWindow = false;

        net = trainWithDropout(net, X(:,trainIdx), y(trainIdx), params.dropoutRate);
        predictions = net(X(:,~trainIdx));
        trueValue = y(~trainIdx);
        valLoss(i) = meanSquaredError(predictions, trueValue);  % Use MSE
    end
    loss = mean(valLoss);
end

function net = trainWithDropout(net, X, y, dropoutRate)
    for i = 1:numel(net.layers)-1
        if isfield(net.layers{i}, 'transferFcn')
            net.layers{i}.transferFcn = @(x) dropoutActivation(x, dropoutRate, net.layers{i}.transferFcn);
        end
    end
    % Disable GUI
    net.trainParam.showWindow = false;
    net = train(net, X, y);
end

function y = dropoutActivation(x, dropoutRate, transferFcn)
    % Apply dropout to the input
    dropoutMask = rand(size(x)) > dropoutRate;
    x = x .* dropoutMask;
    y = feval(transferFcn, x);
end

function mse = meanSquaredError(predictions, trueValue)
    mse = mean((predictions - trueValue).^2);  % Compute mean squared error
end

function net = trainWithoutDropout(net, X, y)
    % Directly use the original activation function instead of Dropout
    for i = 1:numel(net.layers)-1
        if isfield(net.layers{i}, 'transferFcn')
            net.layers{i}.transferFcn = str2func(net.layers{i}.transferFcn);
        end
    end
    % Train the model
    net.trainParam.showWindow = false;  % Disable GUI
    net = train(net, X, y);
end

function [trainMSE, testMSE, trainPCC, testPCC] = looCrossValidateWithoutDropout(net, X, y)
    numSamples = size(X, 2);
    trainErrors = zeros(1, numSamples);
    testErrors = zeros(1, numSamples);
    trainPreds = zeros(1, numSamples);
    testPreds = zeros(1, numSamples);
    
    for i = 1:numSamples
        trainIdx = true(1, numSamples);
        trainIdx(i) = false;

        % Set random seed
        rng(100 + i);  % Set seed for each LOO iteration
        
        % Train the model without Dropout
        net = trainWithoutDropout(net, X(:,trainIdx), y(trainIdx));
        
        % Compute predictions and errors for the training set
        trainPredictions = net(X(:,trainIdx));
        trainErrors(trainIdx) = (trainPredictions - y(trainIdx)).^2;
        trainPreds(trainIdx) = trainPredictions;

        % Compute predictions and errors for the test set
        testPredictions = net(X(:,~trainIdx));
        testErrors(~trainIdx) = (testPredictions - y(~trainIdx)).^2;
        testPreds(~trainIdx) = testPredictions;
    end
    
    % Compute MSE for training and test sets
    trainMSE = mean(trainErrors);
    testMSE = mean(testErrors);

    % Compute PCC for training and test sets using all samples
    trainPCC = calculatePCC(y, trainPreds);  % Use all training predictions to compute PCC
    testPCC = calculatePCC(y, testPreds);  % Use all test predictions to compute PCC
end

function pcc = calculatePCC(trueValues, predictions)
    % Prevent dimensionality issues in input values
    trueValues = trueValues(:);  % Convert data to column vector
    predictions = predictions(:);  % Convert data to column vector
    
    % If variance is zero, all predictions are perfectly correct, return PCC as 1
    if std(trueValues) == 0 || std(predictions) == 0
        pcc = 1;  % If the standard deviation of true values or predictions is zero, predictions are perfectly correct, PCC should be 1
        return;
    end
    
    % Compute Pearson correlation coefficient (PCC)
    pcc = corr(trueValues, predictions);
end

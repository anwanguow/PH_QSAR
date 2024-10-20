% 1. Set file paths
save_model = 'models/reg_laplace_10.mat';  % The saved model file
load_pca_datafile = 'data_pca/reg_laplace_10_pca.csv';  % PCA processed data file
results_file = 'results/reg_laplace_10.csv';  % Path to save predictions and true values as CSV file
fig_file = "results/reg_laplace_10.png";

% 2. Set random seed
rng(2);  % Use the same random seed as the previous LOO loop

% 3. Read PCA processed data
data_pca = readtable(load_pca_datafile);
X_pca = data_pca{:,1:end-1}';
y = data_pca{:,end}';

% 4. Load the saved model
loaded_data = load(save_model);  % Load the saved model and parameters
net = loaded_data.net;  % Extract the saved network model

% 5. Perform Leave-One-Out Cross-Validation and get predictions
[trainMSE, testMSE, trainPCC, testPCC, predictions] = looCrossValidateWithoutDropout(net, X_pca, y);

% 6. Print training and testing MSE and PCC
fprintf('Train MSE: %.4f\n', trainMSE);
fprintf('Train PCC: %.4f\n', trainPCC);
fprintf('Test MSE: %.4f\n', testMSE);
fprintf('Test PCC: %.4f\n', testPCC);

% 7. Restore predictions and true values to the original (pre-normalized) scale
min_val = 223.15;  % Minimum value of the pre-normalized data
max_val = 416;  % Maximum value of the pre-normalized data

% Restore true values and predictions
y_original = y * (max_val - min_val) + min_val;
predictions_original = predictions * (max_val - min_val) + min_val;

% 8. Generate a bar chart comparing true and predicted values for each drug
figure;
bar([y_original' predictions_original'], 'grouped');  % Display true and predicted values side by side
legend('True Values', 'Predicted Values');
title('Comparison of True and Predicted Values for Each Drug');
xlabel('Drug Index');
ylabel('Values');
saveas(gcf, fig_file);  % Save the figure as a PNG file

% 9. Save the restored predictions and true values to a CSV file
output_table = table(y_original', predictions_original', 'VariableNames', {'TrueValues', 'PredictedValues'});
writetable(output_table, results_file);

% Function section (same as before, with predictions output added)
function [trainMSE, testMSE, trainPCC, testPCC, testPreds] = looCrossValidateWithoutDropout(net, X, y)
    numSamples = size(X, 2);
    trainErrors = zeros(1, numSamples);
    testErrors = zeros(1, numSamples);
    trainPreds = zeros(1, numSamples);
    testPreds = zeros(1, numSamples);
    
    for i = 1:numSamples
        trainIdx = true(1, numSamples);
        trainIdx(i) = false;

        % Set random seed
        rng(100 + i);  % Use the same randomness as the previous LOO iteration
        
        % Train the model without Dropout
        net = trainWithoutDropout(net, X(:,trainIdx), y(trainIdx));
        
        % Calculate predictions and errors on the training set
        trainPredictions = net(X(:,trainIdx));
        trainErrors(trainIdx) = (trainPredictions - y(trainIdx)).^2;
        trainPreds(trainIdx) = trainPredictions;

        % Calculate predictions and errors on the test set
        testPredictions = net(X(:,~trainIdx));
        testErrors(~trainIdx) = (testPredictions - y(~trainIdx)).^2;
        testPreds(~trainIdx) = testPredictions;  % Save test predictions
    end
    
    % Calculate MSE for training and test sets
    trainMSE = mean(trainErrors);
    testMSE = mean(testErrors);

    % Calculate PCC for training and test sets across all samples
    trainPCC = calculatePCC(y, trainPreds);  % Calculate PCC using all training predictions
    testPCC = calculatePCC(y, testPreds);  % Calculate PCC using all test predictions
end

function pcc = calculatePCC(trueValues, predictions)
    % Prevent dimension issues with inputs
    trueValues = trueValues(:);  % Convert data to column vector
    predictions = predictions(:);  % Convert data to column vector
    
    % If the variance is zero, it means all predictions are perfectly correct, return PCC as 1
    if std(trueValues) == 0 || std(predictions) == 0
        pcc = 1;  % If the standard deviation of the true or predicted values is zero, it means perfect predictions, PCC should be 1
        return;
    end
    
    % Calculate Pearson correlation coefficient (PCC)
    pcc = corr(trueValues, predictions);
end

function net = trainWithoutDropout(net, X, y)
    % Use the original activation function instead of Dropout
    for i = 1:numel(net.layers)-1
        if isfield(net.layers{i}, 'transferFcn')
            net.layers{i}.transferFcn = str2func(net.layers{i}.transferFcn);
        end
    end
    % Train the model
    net.trainParam.showWindow = false;  % Disable GUI
    net = train(net, X, y);
end

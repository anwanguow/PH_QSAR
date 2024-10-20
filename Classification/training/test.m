% Confusing Matrix
M = [53, 0, 0;
    0, 26, 0;
    2, 0, 48];

disp("MCC ext: ")
mcc_bin_ext(M)

disp("MCC overall: ")
mcc_overall(M)

function mcc = mcc_bin_ext(confMat)
    numClasses = size(confMat, 1);
    TP = zeros(numClasses, 1);
    TN = zeros(numClasses, 1);
    FP = zeros(numClasses, 1);
    FN = zeros(numClasses, 1);
    for i = 1:numClasses
        TP(i) = confMat(i, i);
        FP(i) = sum(confMat(:, i)) - TP(i);
        FN(i) = sum(confMat(i, :)) - TP(i);
        TN(i) = sum(confMat(:)) - (TP(i) + FP(i) + FN(i));
    end
    numerator = sum(TP .* TN) - sum(FP .* FN);
    denominator = sqrt(sum((TP + FP) .* (TP + FN)) * sum((TN + FP) .* (TN + FN)));
    if denominator == 0
        mcc = 0;
    else
        mcc = numerator / denominator;
    end
end

function mcc = mcc_overall(confMat)
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


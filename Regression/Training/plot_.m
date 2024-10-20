results_file = 'results/reg_barcode_40.csv';
data = readtable(results_file);
true_Tg = data.TrueValues;
predicted_Tg = data.PredictedValues;
figure;
scatter(true_Tg, predicted_Tg, 'filled');
hold on;

min_val = min([true_Tg; predicted_Tg]);
max_val = max([true_Tg; predicted_Tg]);
plot([min_val, max_val], [min_val, max_val], 'r--');

xlabel('True Tg');
ylabel('Predicted Tg');
title('Comparison of True and Predicted Tg Values');
legend('Samples', 'y = x Reference Line', 'Location', 'Best');

saveas(gcf, 'results/reg_barcode_40.png');

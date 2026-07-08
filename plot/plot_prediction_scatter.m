function plot_prediction_scatter(yTrue,yPred)
%PLOT_PREDICTION_SCATTER Scatter plot of measured and predicted values.

yTrue = yTrue(:);
yPred = yPred(:);

figure;
scatter(yTrue,yPred,36,'filled');
grid on;
hold on;

minVal = min([yTrue; yPred]);
maxVal = max([yTrue; yPred]);

plot([minVal,maxVal],[minVal,maxVal],'k--','LineWidth',1.2);

xlabel('Measured value');
ylabel('Predicted value');
title('Prediction Scatter Plot');

axis equal;
xlim([minVal,maxVal]);
ylim([minVal,maxVal]);

end

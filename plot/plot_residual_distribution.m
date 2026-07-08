function plot_residual_distribution(yTrue,yPred)
%PLOT_RESIDUAL_DISTRIBUTION Plot residual histogram.

yTrue = yTrue(:);
yPred = yPred(:);

residual = yPred - yTrue;

figure;
histogram(residual,20);
grid on;

xlabel('Residual');
ylabel('Frequency');
title('Residual Distribution');

end

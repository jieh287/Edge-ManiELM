function plot_loss_curve(trainInfo)
%PLOT_LOSS_CURVE Plot training loss curve.

if ~isfield(trainInfo,'lossCurve')
    error('plot_loss_curve:MissingField','trainInfo.lossCurve is required.');
end

figure;
plot(trainInfo.lossCurve,'LineWidth',1.5);
grid on;

xlabel('Epoch');
ylabel('Loss');
title('Training Loss Curve');

end

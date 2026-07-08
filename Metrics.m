function scores = Metrics(yRef, yEst)
%METRICS Compute regression accuracy indicators.
%
%   scores = Metrics(yRef, yEst)
%
%   Inputs
%   ------
%   yRef : measured/reference values
%   yEst : estimated/predicted values
%
%   Outputs
%   -------
%   scores.RMSE      Root mean square error
%   scores.MAE       Mean absolute error
%   scores.r         Pearson correlation coefficient
%   scores.SD        Standard deviation of reference values
%   scores.RPD       Ratio of performance to deviation
%   scores.RPIQ      Ratio of performance to interquartile range
%   scores.MaxRelErr Maximum relative error
%   scores.MinRelErr Minimum relative error

    % Convert inputs to column vectors
    yRef = yRef(:);
    yEst = yEst(:);

    % Basic input checking
    if isempty(yRef) || isempty(yEst)
        error('Metrics:EmptyInput', ...
            'Input vectors must not be empty.');
    end

    if length(yRef) ~= length(yEst)
        error('Metrics:SizeMismatch', ...
            'Reference and estimated vectors must have the same length.');
    end

    % Prediction residual
    err = yEst - yRef;

    % Error-based indicators
    scores.RMSE = sqrt(mean(err .^ 2));
    scores.MAE  = mean(abs(err));

    % Pearson correlation coefficient
    yRef0 = yRef - mean(yRef);
    yEst0 = yEst - mean(yEst);

    denom = norm(yRef0) * norm(yEst0);

    if denom < eps
        scores.r = NaN;
    else
        scores.r = (yRef0' * yEst0) / denom;
    end

    % Dispersion of reference values
    scores.SD = sqrt(mean((yRef - mean(yRef)) .^ 2));

    % RPD and RPIQ
    if scores.RMSE < eps
        scores.RPD  = Inf;
        scores.RPIQ = Inf;
    else
        scores.RPD = scores.SD / scores.RMSE;

        q1 = percentile_no_toolbox(yRef, 25);
        q3 = percentile_no_toolbox(yRef, 75);

        scores.RPIQ = (q3 - q1) / scores.RMSE;
    end

    % Relative error
    den = yRef;
    den(abs(den) < eps) = eps;

    relErr = err ./ den;

    scores.MaxRelErr = max(relErr);
    scores.MinRelErr = min(relErr);
end


function value = percentile_no_toolbox(data, percent)
%PERCENTILE_NO_TOOLBOX Compute percentile without Statistics Toolbox.

    data = sort(data(:));
    n = length(data);

    if n == 1
        value = data;
        return;
    end

    rankPos = 1 + (n - 1) * percent / 100;

    idxLow  = floor(rankPos);
    idxHigh = ceil(rankPos);

    if idxLow == idxHigh
        value = data(idxLow);
    else
        ratio = rankPos - idxLow;
        value = data(idxLow) * (1 - ratio) + data(idxHigh) * ratio;
    end
end
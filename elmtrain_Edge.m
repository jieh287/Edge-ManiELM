function [KAN,LW,TF,TYPE,trainInfo] = elmtrain_Edge(P,T,N,TF,TYPE,numKnots,opts)
%ELMTRAIN_EDGE Train Edge-ManiELM.
%
% This public version keeps only the final mixed quadratic-cubic B-spline
% implementation used in the paper.

if nargin < 2
    error('elmtrain_Edge:Arguments','Not enough input arguments.');
end

if nargin < 3 || isempty(N)
    N = size(P,2);
end

if nargin < 4 || isempty(TF)
    TF = 'kan';
end

if nargin < 5 || isempty(TYPE)
    TYPE = 0;
end

if nargin < 6 || isempty(numKnots)
    numKnots = 10;
end

if nargin < 7
    opts = struct();
end

opts = complete_edge_options(opts);

% Compatibility for the public release.
% These fields replace the old debugging-style splineOrder switch.
if ~isfield(opts,'basisMode')
    opts.basisMode = 'mixed_quadratic_cubic_bspline';
end

if ~isfield(opts,'knotMode') || isnumeric(opts.knotMode)
    opts.knotMode = 'quantile_nonuniform';
end

if ~isfield(opts,'baseMode') || isnumeric(opts.baseMode)
    opts.baseMode = 'silu_input_bypass';
end

if ~isempty(opts.seed)
    rng(opts.seed);
end

if size(P,2) ~= size(T,2)
    error('elmtrain_Edge:SizeMismatch', ...
        'P and T must have the same number of columns.');
end

if TYPE == 1
    T = ind2vec(T);
end

[R,Q] = size(P); %#ok<ASGLU>
[S,~] = size(T); %#ok<ASGLU>

TF = 'kan';

%% Basis construction
% The basis setting is fixed as mixed quadratic and cubic B-splines.

xMin = -1;
xMax =  1;

basisInfo = create_mixed_bspline_info(P,numKnots,xMin,xMax);
Z = build_basis_features(P,basisInfo);

%% Model initialization
% The hidden representation is represented by edge-function coefficients.

KAN = init_edge_model(P,Z,N,numKnots,basisInfo,xMin,xMax,opts);

%% Graph construction
% The graph is only built on training samples.

if opts.useGraph == 1
    graphL = build_graph_laplacian(P,opts);
else
    graphL = [];
end

%% Training loop
% The output layer is solved in closed form at each epoch.

lossCurve   = zeros(opts.maxEpoch,1);
mseCurve    = zeros(opts.maxEpoch,1);
regCurve    = zeros(opts.maxEpoch,1);
smoothCurve = zeros(opts.maxEpoch,1);

for epoch = 1:opts.maxEpoch

    [H,G] = edge_forward(P,Z,KAN); %#ok<ASGLU>

    LW = solve_output_weight( ...
        G,T,opts.lambda,graphL,opts.graphLambda,opts.useGraph);

    Y = (G' * LW)';

    E = Y - T;
    mseLoss = mean(E(:).^2);

    regLoss = opts.weightDecay * mean(KAN.Cflat(:).^2);

    [smoothLoss,gradSmooth] = smooth_regularization_fast( ...
        KAN.Cflat,KAN.N,KAN.R,KAN.numKnots,opts.smoothLambda);

    loss = mseLoss + regLoss + smoothLoss;

    lossCurve(epoch)   = loss;
    mseCurve(epoch)    = mseLoss;
    regCurve(epoch)    = regLoss;
    smoothCurve(epoch) = smoothLoss;

    %% Gradient update
    % Only edge-function coefficients and hidden bias are updated.

    dY = 2 * E / numel(E);

    dG = LW * dY;
    dH = dG(1:N,:);

    gradWeight = 2 * opts.weightDecay * KAN.Cflat / numel(KAN.Cflat);

    dC = dH * Z' + gradWeight + gradSmooth;
    dB = sum(dH,2);

    clipValue = 5;

    dC = max(min(dC,clipValue),-clipValue);
    dB = max(min(dB,clipValue),-clipValue);

    KAN.Cflat = KAN.Cflat - opts.lr * dC;
    KAN.B     = KAN.B     - opts.lr * dB;

    if opts.verbose == 1
        if mod(epoch,20) == 0 || epoch == 1 || epoch == opts.maxEpoch
            fprintf('Epoch %d / %d, loss = %.6f, mse = %.6f\n', ...
                epoch,opts.maxEpoch,loss,mseLoss);
        end
    end
end

%% Final readout
% The final output weights are recomputed after edge parameters are updated.

[~,G] = edge_forward(P,Z,KAN);

LW = solve_output_weight( ...
    G,T,opts.lambda,graphL,opts.graphLambda,opts.useGraph);

%% Training information

trainInfo = struct();

trainInfo.lossCurve   = lossCurve;
trainInfo.mseCurve    = mseCurve;
trainInfo.regCurve    = regCurve;
trainInfo.smoothCurve = smoothCurve;

trainInfo.finalLoss = lossCurve(end);
trainInfo.finalMSE  = mseCurve(end);

trainInfo.opts = opts;

trainInfo.TF = TF;
trainInfo.TYPE = TYPE;

trainInfo.basisMode = 'mixed_quadratic_cubic_bspline';
trainInfo.knotMode  = 'quantile_nonuniform';

trainInfo.baseMode  = 'silu_input_bypass';
trainInfo.useSmooth = 1;

trainInfo.useGraph       = opts.useGraph;
trainInfo.graphLambda    = opts.graphLambda;
trainInfo.graphK         = opts.graphK;
trainInfo.graphUseGeo    = opts.graphUseGeo;
trainInfo.graphUseSpec   = opts.graphUseSpec;
trainInfo.graphNormalize = opts.graphNormalize;

end

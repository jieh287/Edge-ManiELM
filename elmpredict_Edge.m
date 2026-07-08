function Y = elmpredict_Edge(P,KAN,LW,TF,TYPE)
%ELMPREDICT_EDGE Predict with Edge-ManiELM.
%
%   Y = elmpredict_Edge(P,KAN,LW,TF,TYPE)
%
%   Inputs
%   ------
%   P    : normalized input matrix, arranged as features x samples
%   KAN  : trained Edge-ManiELM model structure
%   LW   : output-layer weight matrix
%   TF   : transfer function flag, kept for interface compatibility
%   TYPE : task type, 0 for regression and 1 for classification
%
%   Output
%   ------
%   Y    : predicted output matrix
%
%   Note
%   ----
%   representation adopted in the final Edge-ManiELM model.

%% Input checking
% Only a few necessary fields are checked here.

if nargin < 5
    error('elmpredict_Edge:Arguments','Not enough input arguments.');
end

if ~isfield(KAN,'Cflat')
    error('elmpredict_Edge:ModelError','The model does not contain Cflat.');
end

if ~isfield(KAN,'B')
    error('elmpredict_Edge:ModelError','The model does not contain B.');
end

%% Load basis information
% basisInfo stores the knot vectors and spline component settings.

if isfield(KAN,'basisInfo')
    basisInfo = KAN.basisInfo;
elseif isfield(KAN,'knotInfo')
    basisInfo = KAN.knotInfo;
else
    error('elmpredict_Edge:ModelError','The model does not contain basis information.');
end

% The final model uses the identity aggregation after edge-function mapping.
TF = 'kan'; %#ok<NASGU>

%% Build spline basis features
% The same basis definition used during training is reused here.

Z = build_basis_features(P,basisInfo);

%% Hidden-layer response

numSamples = size(P,2);

H = KAN.Cflat * Z + repmat(KAN.B,1,numSamples);

%% Readout matrix
% A SiLU input bypass is appended to preserve smooth low-order information.

G = [H; silu(P)];

%% Output prediction
% The output-layer weights are obtained from the training stage.

Y = (G' * LW)';

%% Classification output conversion
% This branch is kept for interface completeness.

if TYPE == 1

    tempY = zeros(size(Y));

    for i = 1:size(Y,2)
        [~,idx] = max(Y(:,i));
        tempY(idx,i) = 1;
    end

    Y = vec2ind(tempY);
end

end


function Z = build_basis_features(P,basisInfo)
%BUILD_BASIS_FEATURES Build mixed B-spline basis response matrix.
%
% Each input dimension is expanded independently.
% The final basis matrix is stacked by feature dimension.

[R,Q] = size(P);
Ktotal = basisInfo.numKnots;

Z = zeros(R*Ktotal,Q);

row = 1;

for r = 1:R

    % Limit the input values to the normalized modeling interval.
    x = P(r,:);
    x = max(min(x,basisInfo.x_max),basisInfo.x_min);

    B_all = [];

    % The public model contains two spline components:
    % quadratic B-spline and cubic B-spline.
    for cidx = 1:length(basisInfo.components)

        comp = basisInfo.components{cidx};

        Bmat = bspline_basis_matrix_with_vector( ...
            x, ...
            comp.K, ...
            comp.degree, ...
            comp.knotVectors{r}, ...
            basisInfo.x_max);

        B_all = [B_all; Bmat]; %#ok<AGROW>
    end

    % This check avoids silent dimension inconsistency.
    if size(B_all,1) ~= Ktotal
        error('Basis feature number mismatch.');
    end

    Z(row:row+Ktotal-1,:) = B_all;
    row = row + Ktotal;
end

end


function Bmat = bspline_basis_matrix_with_vector(x,nBasis,degree,t,x_max)
%BSPLINE_BASIS_MATRIX_WITH_VECTOR Evaluate B-spline basis functions.
%
% The implementation follows the Cox-de Boor recursion.

x = x(:)';
Q = length(x);

%% Zeroth-degree basis
% The first step assigns each sample to a knot interval.

B = zeros(nBasis + degree,Q);

for i = 1:(nBasis + degree)
    B(i,:) = double(x >= t(i) & x < t(i+1));
end

%% Recursive basis construction

for p = 1:degree

    Bnew = zeros(nBasis + degree - p,Q);

    for i = 1:(nBasis + degree - p)

        denom1 = t(i+p) - t(i);
        denom2 = t(i+p+1) - t(i+1);

        term1 = zeros(1,Q);
        term2 = zeros(1,Q);

        if denom1 > 0
            term1 = ((x - t(i)) ./ denom1) .* B(i,:);
        end

        if denom2 > 0
            term2 = ((t(i+p+1) - x) ./ denom2) .* B(i+1,:);
        end

        Bnew(i,:) = term1 + term2;
    end

    B = Bnew;
end

Bmat = B(1:nBasis,:);

%% Right boundary correction
% The right endpoint is assigned to the last basis function.

rightIdx = abs(x - x_max) <= 1e-12;

if any(rightIdx)
    Bmat(:,rightIdx) = 0;
    Bmat(end,rightIdx) = 1;
end

%% Basis normalization
% This small step improves numerical stability near knot boundaries.

s = sum(Bmat,1);
idx = s > 0;

Bmat(:,idx) = Bmat(:,idx) ./ s(idx);

end


function Y = silu(X)
%SILU SiLU activation function.
%
% SiLU(x) = x / (1 + exp(-x)).

Y = X ./ (1 + exp(-X));

end
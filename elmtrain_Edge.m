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

%% Input checking

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

Z = build_basis_features(P,basisInfo);

%% Forward calculation
% The hidden response and readout matrix are constructed by the shared
% forward function in the src folder.

[~,G] = edge_forward(P,Z,KAN);

%% Output prediction
% The output-layer weights are obtained from the training stage.

Y = (G' * LW)';

%% Classification output conversion

if TYPE == 1

    tempY = zeros(size(Y));

    for i = 1:size(Y,2)
        [~,idx] = max(Y(:,i));
        tempY(idx,i) = 1;
    end

    Y = vec2ind(tempY);
end

end

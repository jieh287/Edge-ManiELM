function opts = edge_default_config()
%EDGE_DEFAULT_CONFIG Default parameters for Edge-ManiELM.
%
% This file is mainly used to keep the project structure clear.

opts = struct();

%% Training settings

opts.maxEpoch    = 100;
opts.lr          = 0.01;
opts.lambda      = 1e-4;
opts.weightDecay = 1e-5;

%% Edge-function settings
% The public release keeps only the final basis design.

opts.basisMode = 'mixed_quadratic_cubic_bspline';
opts.knotMode  = 'quantile_nonuniform';
opts.numKnots  = 10;

%% Smoothness setting

opts.useSmooth    = 1;
opts.smoothLambda = 1e-4;

%% Bypass setting

opts.baseMode  = 'silu_input_bypass';
opts.baseDecay = 1e-5;

%% Graph setting

opts.useGraph       = 1;
opts.graphK         = 5;
opts.graphUseGeo    = 1;
opts.graphUseSpec   = 1;
opts.graphLambda    = 1e-1;
opts.graphNormalize = 1;
opts.graphCoord     = [];

%% Display setting

opts.verbose = 0;
opts.seed    = [];

end

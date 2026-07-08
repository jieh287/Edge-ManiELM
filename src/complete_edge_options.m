function opts = complete_edge_options(opts)
%COMPLETE_EDGE_OPTIONS Fill missing Edge-ManiELM options.

if nargin < 1 || isempty(opts)
    opts = struct();
end

if ~isfield(opts,'maxEpoch');      opts.maxEpoch = 100; end
if ~isfield(opts,'lr');            opts.lr = 0.01; end
if ~isfield(opts,'lambda');        opts.lambda = 1e-4; end
if ~isfield(opts,'weightDecay');   opts.weightDecay = 1e-5; end
if ~isfield(opts,'smoothLambda');  opts.smoothLambda = 1e-4; end
if ~isfield(opts,'numKnots');      opts.numKnots = 10; end
if ~isfield(opts,'graphCoord');    opts.graphCoord = []; end
if ~isfield(opts,'verbose');       opts.verbose = 0; end
if ~isfield(opts,'seed');          opts.seed = []; end

opts.basisMode = 'mixed_quadratic_cubic_bspline';
opts.knotMode  = 'quantile_nonuniform';

opts.useSmooth = 1;
opts.baseMode  = 'silu_input_bypass';

opts.useGraph       = 1;
opts.graphK         = 5;
opts.graphUseGeo    = 1;
opts.graphUseSpec   = 1;
opts.graphLambda    = 1e-1;
opts.graphNormalize = 1;

end

function KAN = init_edge_model(P,Z,N,numKnots,basisInfo,xMin,xMax,opts)
%INIT_EDGE_MODEL Initialize Edge-ManiELM model parameters.

[R,~] = size(P);
D = size(Z,1);

scaleC = sqrt(2 / D);

Cflat = randn(N,D) * scaleC;
B = randn(N,1) * 0.1;

KAN = struct();

KAN.N = N;
KAN.R = R;
KAN.D = D;

KAN.numKnots = numKnots;

KAN.basisMode = opts.basisMode;
KAN.knotMode  = opts.knotMode;

KAN.basisInfo = basisInfo;
KAN.knotInfo  = basisInfo;

KAN.Cflat = Cflat;
KAN.B     = B;
KAN.Wbase = zeros(N,R);

KAN.x_min = xMin;
KAN.x_max = xMax;

KAN.baseMode  = opts.baseMode;
KAN.useSmooth = 1;

KAN.useGraph       = opts.useGraph;
KAN.graphLambda    = opts.graphLambda;
KAN.graphK         = opts.graphK;
KAN.graphUseGeo    = opts.graphUseGeo;
KAN.graphUseSpec   = opts.graphUseSpec;
KAN.graphNormalize = opts.graphNormalize;

end

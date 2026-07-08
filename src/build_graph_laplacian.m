function L = build_graph_laplacian(P,opts)
%BUILD_GRAPH_LAPLACIAN Build spatial-spectral graph Laplacian.

Q = size(P,2);

if Q <= 1
    L = zeros(Q,Q);
    return;
end

useGeo  = opts.graphUseGeo;
useSpec = opts.graphUseSpec;

D_comb = zeros(Q,Q);
W = ones(Q,Q);

%% Spatial affinity

if useGeo == 1

    coord = opts.graphCoord;

    if isempty(coord)
        useGeo = 0;
    else
        if size(coord,1) == 2 && size(coord,2) == Q
            coordUse = coord;
        elseif size(coord,2) == 2 && size(coord,1) == Q
            coordUse = coord';
        else
            useGeo = 0;
        end
    end

    if useGeo == 1

        coordXY = coordinate_to_meter(coordUse);

        D_geo = pairwise_distance(coordXY);
        sigmaGeo = median_positive(D_geo);

        W_geo = exp(-(D_geo.^2) ./ (2*sigmaGeo^2 + eps));

        W = W .* W_geo;
        D_comb = D_comb + D_geo ./ (sigmaGeo + eps);
    end
end

%% Spectral affinity

if useSpec == 1

    D_spec = pairwise_distance(P);
    sigmaSpec = median_positive(D_spec);

    W_spec = exp(-(D_spec.^2) ./ (2*sigmaSpec^2 + eps));

    W = W .* W_spec;
    D_comb = D_comb + D_spec ./ (sigmaSpec + eps);
end

if useGeo == 0 && useSpec == 0
    L = zeros(Q,Q);
    return;
end

W(1:Q+1:end) = 0;
D_comb(1:Q+1:end) = inf;

%% KNN sparsification

K = min(max(round(opts.graphK),1),Q-1);

W_knn = zeros(Q,Q);

for i = 1:Q
    [~,idx] = sort(D_comb(i,:),'ascend');
    idx = idx(1:K);
    W_knn(i,idx) = W(i,idx);
end

W = max(W_knn,W_knn');
W(1:Q+1:end) = 0;

D = diag(sum(W,2));
L = D - W;

if opts.graphNormalize == 1
    d = diag(D);
    D_inv_sqrt = diag(1 ./ sqrt(d + eps));
    L = D_inv_sqrt * L * D_inv_sqrt;
end

end

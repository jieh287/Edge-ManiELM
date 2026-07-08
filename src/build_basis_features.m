function Z = build_basis_features(P,basisInfo)
%BUILD_BASIS_FEATURES Build mixed B-spline basis response matrix.

[R,Q] = size(P);
Ktotal = basisInfo.numKnots;

Z = zeros(R*Ktotal,Q);

rowStart = 1;

for r = 1:R

    x = P(r,:);
    x = max(min(x,basisInfo.x_max),basisInfo.x_min);

    B_all = [];

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

    if size(B_all,1) ~= Ktotal
        error('build_basis_features:SizeMismatch', ...
            'Basis feature number mismatch.');
    end

    Z(rowStart:rowStart+Ktotal-1,:) = B_all;

    rowStart = rowStart + Ktotal;
end

end

function basisInfo = create_mixed_bspline_info(P,numKnots,xMin,xMax)
%CREATE_MIXED_BSPLINE_INFO Create mixed quadratic and cubic B-spline basis.

[R,~] = size(P);

K2 = floor(numKnots / 2);
K3 = numKnots - K2;

basisInfo = struct();

basisInfo.numKnots = numKnots;

basisInfo.basisMode = 'mixed_quadratic_cubic_bspline';
basisInfo.knotMode  = 'quantile_nonuniform';

basisInfo.x_min = xMin;
basisInfo.x_max = xMax;

basisInfo.componentDegrees = [2,3];
basisInfo.componentKnots   = [K2,K3];
basisInfo.components       = cell(2,1);

for cidx = 1:2

    degree = basisInfo.componentDegrees(cidx);
    Kc     = basisInfo.componentKnots(cidx);

    comp = struct();

    comp.degree = degree;
    comp.K = Kc;
    comp.knotVectors = cell(R,1);

    for r = 1:R

        x = P(r,:);
        x = x(isfinite(x));
        x = max(min(x,xMax),xMin);

        comp.knotVectors{r} = make_open_knot_vector_quantile( ...
            x,Kc,degree,xMin,xMax);
    end

    basisInfo.components{cidx} = comp;
end

end

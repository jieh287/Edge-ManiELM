function Bmat = bspline_basis_matrix_with_vector(x,nBasis,degree,t,xMax)
%BSPLINE_BASIS_MATRIX_WITH_VECTOR Evaluate B-spline basis functions.

x = x(:)';
Q = length(x);

B = zeros(nBasis + degree,Q);

for i = 1:(nBasis + degree)
    B(i,:) = double(x >= t(i) & x < t(i+1));
end

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

rightIdx = abs(x - xMax) <= 1e-12;

if any(rightIdx)
    Bmat(:,rightIdx) = 0;
    Bmat(end,rightIdx) = 1;
end

s = sum(Bmat,1);
idx = s > 0;

Bmat(:,idx) = Bmat(:,idx) ./ s(idx);

end

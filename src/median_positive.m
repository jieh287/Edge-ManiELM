function sigma = median_positive(D)
%MEDIAN_POSITIVE Median of positive finite values.

v = D(:);
v = v(isfinite(v) & v > 0);

if isempty(v)
    sigma = 1;
else
    sigma = median(v);

    if sigma <= 0 || ~isfinite(sigma)
        sigma = 1;
    end
end

end

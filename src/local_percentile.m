function qv = local_percentile(x,p)
%LOCAL_PERCENTILE Percentile calculation without toolbox dependency.

x = sort(x(:));
n = numel(x);

qv = zeros(size(p));

for i = 1:numel(p)

    if n == 1
        qv(i) = x;
    else
        pos = 1 + (n - 1) * p(i) / 100;

        lo = floor(pos);
        hi = ceil(pos);

        if lo == hi
            qv(i) = x(lo);
        else
            w = pos - lo;
            qv(i) = (1 - w) * x(lo) + w * x(hi);
        end
    end
end

end

function Y = silu(X)
%SILU SiLU activation function.

Y = X ./ (1 + exp(-X));

end

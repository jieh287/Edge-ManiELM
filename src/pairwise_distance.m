function D = pairwise_distance(X)
%PAIRWISE_DISTANCE Pairwise Euclidean distance.

Q = size(X,2);
D = zeros(Q,Q);

for i = 1:Q
    diffX = X - X(:,i);
    D(i,:) = sqrt(sum(diffX.^2,1));
end

end

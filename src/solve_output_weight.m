function LW = solve_output_weight(G,T,lambda,graphL,graphLambda,useGraph)
%SOLVE_OUTPUT_WEIGHT Closed-form readout solution.

M = size(G,1);

A = G * G' + lambda * eye(M);

if useGraph == 1 && ~isempty(graphL) && graphLambda > 0
    A = A + graphLambda * (G * graphL * G');
end

LW = A \ (G * T');

end

function [H,G] = edge_forward(P,Z,KAN)
%EDGE_FORWARD Forward propagation of Edge-ManiELM.

Q = size(P,2);

H = KAN.Cflat * Z + repmat(KAN.B,1,Q);

% SiLU bypass preserves low-order input information.
G = [H; silu(P)];

end

function [smoothLoss,gradSmooth] = smooth_regularization_fast(Cflat,N,R,K,smoothLambda)
%SMOOTH_REGULARIZATION_FAST Second-order coefficient smoothing.

gradSmooth = zeros(size(Cflat));

if K < 3 || smoothLambda <= 0
    smoothLoss = 0;
    return;
end

C3 = reshape(Cflat,[N,K,R]);
C3 = permute(C3,[1,3,2]);

D2 = C3(:,:,3:K) - 2*C3(:,:,2:K-1) + C3(:,:,1:K-2);

count = numel(D2);

smoothLoss = smoothLambda * sum(D2(:).^2) / count;

G = 2 * smoothLambda * D2 / count;

grad3 = zeros(size(C3));

grad3(:,:,1:K-2) = grad3(:,:,1:K-2) + G;
grad3(:,:,2:K-1) = grad3(:,:,2:K-1) - 2 * G;
grad3(:,:,3:K)   = grad3(:,:,3:K)   + G;

grad3 = permute(grad3,[1,3,2]);

gradSmooth = reshape(grad3,[N,R*K]);

end

function [P, L] = ComputeLikelihood(X, gmm, M)
    
    T = length(X);
    D = size(X, 2);
    B = zeros(T, M);
    P = zeros(T, M);
    
    for m=1:M
        B(:,m) = (exp( (-1/2) * ...
            sum( (((X - repmat(gmm.means(:, m)', T, 1)).^2) ./ repmat(diag(gmm.cov(:,:,m))', T, 1)), 2) ...
            )) / ...
            ((2*pi)^(D/2) * sqrt(prod(diag(gmm.cov(:,:,m))')));
    end
        
    % denominator of p(m|x,theta)
    denom_p = sum(repmat(gmm.weights, T, 1) .* B, 2);
    % log likelihood
    L = sum(log(denom_p));

    for m=1:M
        P(:,m) = (repmat(gmm.weights(1,m), T, 1) .* B(:,m)) ./ denom_p; 
    end


end
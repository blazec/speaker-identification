function gmm = UpdateParameters(gmm, X, P, M)
    
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture
    T = length(X);
    D = size(X, 2);
    gmm.weights = sum(P, 1) / T;
    gmm.means = (P' * X)' ./ repmat(sum(P, 1), D, 1);
    cov = (P' * (X.^2))' ./ repmat(sum(P, 1), D, 1) - gmm.means.^2;

    for m=1:M
        gmm.cov(:,:,m) = diag(cov(:,m)); 
    end
end
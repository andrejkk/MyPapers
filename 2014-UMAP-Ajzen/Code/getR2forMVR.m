% It computes R^2 of multivariate regression based on 
% the output of [beta,sig,resid,vars,loglik] = mvregress(X,Y)
function R2 = getR2forMVR(Y, resid)
    n = size(resid, 1);
    SSE = norm(resid).^2;
    TSS = norm(Y - repmat(mean(Y), n, 1))^2;
    R2 = 1 - SSE/TSS;

end
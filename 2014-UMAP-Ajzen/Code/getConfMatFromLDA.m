% Get conf mat from LDA
% Leave one out scheme
function confMat = getConfMatFromLDA(X, gC)

    clsInds = sort(unique(gC));
    numOfCls = numel(clsInds);
    numOfItms = size(X, 1);
    
    clsIndMax = max(gC); % Compute inverse clss inds
    invClsInds = zeros(1, clsIndMax);
    for (jj=1:numOfCls)
        invClsInds(clsInds(jj)) = find(clsInds == clsInds(jj));
    end
    
    % Leave one out
    confMat = zeros(numOfCls);
    
    for (ii = 1:numOfItms)
        currLst = [1:ii-1, ii+1:numOfItms];
        currX = X(currLst, :);
        currGC = gC(currLst);
        currItem = X(ii, :);
        trueClsIndPos = invClsInds(gC(ii));
        [class,err,POSTERIOR,logp,coeff] = classify(currItem, currX, currGC);
        estClsIndPos = invClsInds(class);
        confMat(trueClsIndPos, estClsIndPos) = confMat(trueClsIndPos, estClsIndPos) + 1;
    end
    
end
% It computed separability according to Fisher discriminat analysis
% See http://en.wikipedia.org/wiki/Fisher_discriminant_analysis
%
function sep = computeSeparability(dtX, grpG)

    clsIds = unique(grpG);
    numCls = numel(clsIds);
    numF = size(dtX, 2);
    covAll = cov(dtX);
    
    % Get data by classes & compute class moments
    dtXbyC = cell(1, numCls);
    muC = cell(1, numCls);
    covC = cell(1, numCls);
    muSum = zeros(1, numF);
    for (ii=1:numCls)
        dtXbyC{ii} = dtX(grpG == clsIds(ii), :);
        muC{ii} = mean(dtXbyC{ii}, 1);
        covC{ii} = cov(dtXbyC{ii});
        muSum = muSum + muC{ii}; 
    end
    muAll = muSum/numCls;

    % Compute Fisher vairance
    covB = zeros(numF);
    for (ii=1:numCls)
        covB = covB + (muC{ii}-muAll)'*(muC{ii}-muAll);
    end
    covB = covB/numCls;
    
    % Compute eignevalue
    fishA = pinv(covAll)*covB;
    lambdaVec = eig(fishA);
    sep = max(lambdaVec);
end

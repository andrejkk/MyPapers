% It computes precission (P), recall (R), F- measure (F) and accuracy (A) 
% for each class of a confusion matrix
% @param confusionMat: confusion matrix, eacn row for true class, each
% column for estimated class
% @return prfaMat: each row for i-th class [P_i, R_i, F_i, A_i]
function [prfaMat, AccAll] = computePrecRecFmeasAcc(confusionMat)

    numOfCls = size(confusionMat, 1);
    prfaMat = zeros(numOfCls, 4);
    
    % compute [P_i, R_i, F_i, A_i] for each class
    for (clsInd = 1:numOfCls)
       
        TP = confusionMat(clsInd, clsInd);
        FP = sum([confusionMat(1:clsInd-1, clsInd); confusionMat(clsInd+1:end, clsInd)]);
        FN = sum([confusionMat(clsInd, 1:clsInd-1), confusionMat(clsInd, clsInd+1:end)]);
        TN = sum(sum(confusionMat)) - (TP + FP + FN);
        
        P = TP/(TP + FP);
        R = TP/(TP + FN);
        F = 2*P*R/(P + R);
        A = (TP + TN)/(TP + TN + FP + FN);
        
        prfaMat(clsInd, :) = [P, R, F, A];
    end

    AccAll = sum(diag(confusionMat))/sum(sum(confusionMat));
    
end
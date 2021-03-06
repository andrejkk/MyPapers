%% UMAP 14: Ajzen model fitting

% X = DATA(1:101,1:99);
% [M,N] = size(X);
% y = DATA(1:101,100);
% %X = [ones(M,1) X]; - this thing is done automatically
% r = randperm(size(X, 1));
% trainX = X(r(1:70,:), :);
% trainY = y(r(1:70,:), :);
% testX = X(r(71:end,:),:);
% testY = y(r(71:end,:),:);
% model = LinearModel.fit(trainX, trainY);
% 
% generalizationError = mse(model.predict(testX) - testY)


%% Data
fileNameBase = 'MovieSelectionSurvey_V02';
fullFileNameXLS = fullfile('../Data', strcat(fileNameBase, '.xlsx'));
fullFileNameMat = fullfile('../Data', strcat(fileNameBase, '.mat'));
%[dataRawX,txt,rawRawX] = xlsread(fullFileNameXLS, 'Responses');
%save(fullFileNameMat, 'dataRawX', 'rawRawX');
load(fullFileNameMat, 'dataRawX', 'rawRawX');

% Var names
varNames = rawRawX(1, :);


% Encode categorical vars: C38 - C48:
% C38: alone = 1, Company = 2
for(ii=1:size(dataRawX, 1))
   if (strcmp(strtrim(rawRawX(ii+1, 38)), 'Alone'))
       dataRawX(ii, 38) = 1;
   elseif (strcmp(strtrim(rawRawX(ii+1, 38)), 'Company'))
       dataRawX(ii, 38) = 2;
   end
end

% C39: Premiere = 1, Later = 2
for(ii=1:size(dataRawX, 1))
   if (strcmp(strtrim(rawRawX(ii+1, 39)), 'Premiere'))
       dataRawX(ii, 39) = 1;
   elseif (strcmp(strtrim(rawRawX(ii+1, 39)), 'Later'))
       dataRawX(ii, 39) = 2;
   else
       dataRawX(ii, 39) = -1;
   end
end

% C40 - C48: No = 1, Yes = 2
for (jj = 40:48)
    for(ii=1:size(dataRawX, 1))
        if (strcmp(strtrim(rawRawX(ii+1, jj)), 'No'))
            dataRawX(ii, jj) = 1;
        elseif (strcmp(strtrim(rawRawX(ii+1, jj)), 'Yes'))
            dataRawX(ii, jj) = 2;
        end
    end
end



% Select genres and users
userSel = find(dataRawX(1:35, 2));
dataX = dataRawX(userSel, :);

% Select dependent var
% gDramaCountInd = 75;
% gDramaAverInd = 73;
% gActionCountInd = 84;
% gActionAverInd = 82;
% gComedyCountInd = 78;
% gComedyAverInd = 76;
gDramaQ = 79;
gActionQ = 94;
gComedyQ = 84;

% gCountSel = [gDramaCountInd, gActionCountInd, gComedyCountInd];
% gAvgSel = [gDramaAverInd, gActionAverInd, gComedyAverInd];


% Select predictors
attitudeCogSel = 4:11;
attitudeEmotSel = 12:24;
attitudeBehASel = 25:37;
attitudeBehBSel = 38:49;
attitudeSel = [attitudeCogSel, attitudeEmotSel, attitudeBehASel, attitudeBehBSel];
normsSel = 50:53;
controlSel = 54:60;

% X = dataX(:, [attitudeBehASel, attitudeBehBSel]);
% m = 5;
% [lambda,psi,T,stats,F] = factoran(X,m);
% Trsh = 0.6;
% attitudeBehSel = find(sum(lambda.^2, 2)>0.6)'+24;
attitudeBehSel = [25, 26, 27, 28, 29, 30, 31, 35, 42, 49]; % T = 0.65, 31 added

%% ========================================================================
% Criteria variables
% score variables
sumVotes3g = dataX(:, 77) + dataX(:, 92) + dataX(:, 82);
gS_drama = dataX(:, 77)./(3*dataX(:, 69));
gS_action = dataX(:, 92)./(3*dataX(:, 69));
gS_comedy = dataX(:, 82)./(3*dataX(:, 69));
gS_theRest = sumVotes3g ./(3*dataX(:, 69));

gC_drama = dataX(:, 75);
gC_action = dataX(:, 90);
gC_comedy = dataX(:, 80);
gC_List = [gC_drama, gC_action, gC_comedy];
[C,I] = max(gC_List, [], 2);
gC = I;

% Tu poizkusi upo�tevat �e �tevilo votov: P[like] = f(r_avg, count)
%gC_ListM = [gC_drama.*dataX(:, 77), gC_action.*dataX(:, 92), gC_comedy.*dataX(:, 82)];
%[C,IM] = max(gC_ListM, [], 2);
%gCM = IM;

%% Model: regresion, LDA
outFormat = '%-6.2f';

% Prepare data
maxNumCoeffs = 50;
rowLabelsRegStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsRegStack{ii} = ['$\beta_{', num2str(ii-1), '}$'];
end

rowLabelsLdaStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsLdaStack{ii} = ['$w_{', num2str(ii), '}$'];
end

% Regress, Criteria var: gS
gS_List = [gS_drama, gC_action, gC_comedy];

% -------------------------------------------------------------------------
% Regression models
currSel = attitudeCogSel;

dtX = dataX(:, currSel);
xLen = size(dtX, 2);
currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'attitudeCog_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Multivariate
[nobs,nregions] = size(gS_List);
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
attitudeCog_R2 = getR2forMVR(Y, resid);

% mdl = LinearModel.fit(dtX,Y);

% Scores
X = [ones(nobs, 1), dtX];
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
attitudeCogScores = X*regCoeffs';

% Single
X = [ones(nobs, 1), dataX(:, currSel)];
[b_drama,bint,r,rint,stats_drama] = regress(gS_drama, X);
[b_action,bint,r,rint,stats_action] = regress(gC_action, X);
[b_comedy,bint,r,rint,stats_comedy] = regress(gC_comedy, X);

% Export multiple regress coeffs
outFN = 'attitudeCog_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% LDA, Criteria var: gc
X = dataX(:, currSel);
% %cls = ClassificationDiscriminant.fit(X, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(X, 2)), X, gC);
ldaCoeffs = zeros(3, size(X, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

attitudeCog_Sep = computeSeparability(X, gC);
confMat = getConfMatFromLDA(X, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'attitudeCog_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% -------------------------------------------------------------------------
% Regression models
currSel = attitudeEmotSel;

dtX = dataX(:, currSel);
xLen = size(dtX, 2);
currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'attitudeEmot_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% Multivariate
[nobs,nregions] = size(gS_List);
dtX = dataX(:, currSel);
xLen = size(dtX, 2);
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
attitudeEmot_R2 = getR2forMVR(Y, resid);

% Scores
X = [ones(nobs, 1), dtX];
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
attitudeEmotScores = X*regCoeffs';


% Single
[b_drama,bint,r,rint,stats_drama] = regress(gS_drama, X);
[b_action,bint,r,rint,stats_action] = regress(gC_action, X);
[b_comedy,bint,r,rint,stats_comedy] = regress(gC_comedy, X);

% Export multiple regress coeffs
outFN = 'attitudeEmot_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% LDA, Criteria var: gc
X = dataX(:, currSel);
% %cls = ClassificationDiscriminant.fit(X, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(X, 2)), X, gC);
ldaCoeffs = zeros(3, size(X, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

attitudeEmot_Sep = computeSeparability(X, gC);
confMat = getConfMatFromLDA(X, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'attitudeEmot_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% -------------------------------------------------------------------------
% Regression models
currSel = attitudeBehSel;


dtX = dataX(:, currSel);
xLen = size(dtX, 2);
currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'attitudeBeh_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Multivariate
[nobs,nregions] = size(gS_List);
dtX = dataX(:, currSel);
xLen = size(dtX, 2);
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
attitudeBeh_R2 = getR2forMVR(Y, resid);

% Scores
X = [ones(nobs, 1), dtX];
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
attitudeBehScores = X*regCoeffs';

% Single
[b_drama,bint,r,rint,stats_drama] = regress(gS_drama, X);
[b_action,bint,r,rint,stats_action] = regress(gC_action, X);
[b_comedy,bint,r,rint,stats_comedy] = regress(gC_comedy, X);

% Export multiple regress coeffs
outFN = 'attitudeBeh_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% LDA, Criteria var: gc
X = dataX(:, currSel);
% %cls = ClassificationDiscriminant.fit(X, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(X, 2)), X, gC);
ldaCoeffs = zeros(3, size(X, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

attitudeBeh_Sep = computeSeparability(X, gC);
confMat = getConfMatFromLDA(X, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'attitudeBeh_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% -------------------------------------------------------------------------
% Regression models
currSel = normsSel;


dtX = dataX(:, currSel);
xLen = size(dtX, 2);
currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'norms_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Multivariate
[nobs,nregions] = size(gS_List);
dtX = dataX(:, currSel);
xLen = size(dtX, 2);
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
norms_R2 = getR2forMVR(Y, resid);

% Scores
X = [ones(nobs, 1), dtX];
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
normsScores = X*regCoeffs';

% Single
[b_drama,bint,r,rint,stats_drama] = regress(gS_drama, X);
[b_action,bint,r,rint,stats_action] = regress(gC_action, X);
[b_comedy,bint,r,rint,stats_comedy] = regress(gC_comedy, X);

% Export multiple regress coeffs
outFN = 'norms_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% LDA, Criteria var: gc
X = dataX(:, currSel);
% %cls = ClassificationDiscriminant.fit(X, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(X, 2)), X, gC);
ldaCoeffs = zeros(3, size(X, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

norms_Sep = computeSeparability(X, gC);
confMat = getConfMatFromLDA(X, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'norms_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% -------------------------------------------------------------------------
% Regression models
currSel = controlSel;

xLen = size(dtX, 2);
dtX = dataX(:, currSel);
xLen = size(dtX, 2);
currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'control_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Multivariate
[nobs,nregions] = size(gS_List);
dtX = dataX(:, currSel);
xLen = size(dtX, 2);
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
control_R2 = getR2forMVR(Y, resid);

% Scores
X = [ones(nobs, 1), dtX];
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
controlScores = X*regCoeffs';

% Single
[b_drama,bint,r,rint,stats_drama] = regress(gS_drama, X);
[b_action,bint,r,rint,stats_action] = regress(gC_action, X);
[b_comedy,bint,r,rint,stats_comedy] = regress(gC_comedy, X);

% Export multiple regress coeffs
outFN = 'control_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% LDA, Criteria var: gc
X = dataX(:, currSel);
% %cls = ClassificationDiscriminant.fit(X, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(X, 2)), X, gC);
ldaCoeffs = zeros(3, size(X, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

control_Sep = computeSeparability(X, gC);
confMat = getConfMatFromLDA(X, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'control_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

%% Top regression model
% Multivariate
[nobs,nregions] = size(gS_List);
dtX = [attitudeCogScores, attitudeEmotScores, attitudeBehScores, ...
    normsScores, controlScores];
xLen = size(dtX, 2);


currCorr = corr(dtX);
maxCorr = max(max(abs(currCorr - eye(xLen))));

outFN = 'top_Corr.tex';
columnLabels = {};
rowLabels = {};
matrix2latex(currCorr, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), ...
        [dtX(j, :), zeros(1, 2*xLen); ...
         zeros(1, xLen), dtX(j, :), zeros(1, xLen);...
         zeros(1, 2*xLen), dtX(j, :)]]; 
end
Y = gS_List;
[beta,sig,resid,vars,loglik] = mvregress(X,Y);
top_R2 = getR2forMVR(Y, resid);

% Export multiple regress coeffs
outFN = 'top_MvRegress.tex';
columnLabels = rowLabelsRegStack(1:xLen+1);
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [[beta(1), beta(4:4+xLen-1)']; [beta(2), beta(4+xLen:3+2*xLen)']; [beta(3), beta(4+2*xLen:3+3*xLen)']];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% LDA, Criteria var: gc
% cls = ClassificationDiscriminant.fit(dtX, gC);
% ldaCoeffs = cls.Mu;

[class,err,POSTERIOR,logp,coeff] = classify(ones(1, size(dtX, 2)), dtX, gC);
ldaCoeffs = zeros(3, size(dtX, 2));
ldaCoeffs(1, :) = coeff(1,2).linear';
ldaCoeffs(2, :) = coeff(1,3).linear';
ldaCoeffs(3, :) = coeff(2,3).linear';

tot_Sep = computeSeparability(dtX, gC);
confMat = getConfMatFromLDA(dtX, gC);
[prfaMat, AccAll] = computePrecRecFmeasAcc(confMat);

% Export multiple regress coeffs
outFN = 'top_LDA.tex';
columnLabels = rowLabelsLdaStack(1:xLen);
rowLabels = {'Drama/Action', 'Drama/Comedy', 'Action/Comedy'}; 
matrix2latex(ldaCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% All
all_R2 = [attitudeCog_R2, attitudeEmot_R2, attitudeBeh_R2, norms_R2, control_R2];
all_Sep = [attitudeCog_Sep, attitudeEmot_Sep, attitudeBeh_Sep, norms_Sep, control_Sep];

% Export all stufs
outFN = 'all_R2andSeps.tex';
columnLabels = {'Attr: cognitive', 'Attr: emotions', 'Attr: behavioral', 'Norms', 'Control'};
rowLabels = {'$R^2$', 'Separability'}; 
matrix2latex([all_R2; all_Sep], outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


%% Factor analysis
% rot: varimax
outFormat = '%-6.2f';

% Prepare data
maxNumCoeffs = 50;
rowLabelsFactorAnStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsFactorAnStack{ii} = ['$Q_{', num2str(ii), '}$'];
end
colLabelsFactorAnStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    colLabelsFactorAnStack{ii} = ['$F_{', num2str(ii), '}$'];
end

% Atttude Cognitive
currSel = attitudeCogSel;
outFN = 'attitudeCog_FactorAn.tex';

X = dataX(:, currSel);
m = 4;
[lambda,psi,T,stats,F] = factoran(X, m); %, 'rotate', 'none');

Comm = sum(lambda.^2, 2);
Uniq = 1 - Comm;
factorVars = sum(lambda.^2);

columnLabels = colLabelsFactorAnStack(1:size(lambda, 2));
rowLabels = rowLabelsFactorAnStack(1:size(lambda, 1));
matrix2latex(lambda, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% Atttude Emotive
currSel = attitudeEmotSel;
outFN = 'attitudeEmot_FactorAn.tex';

X = dataX(:, currSel);
m = 4;
[lambda,psi,T,stats,F] = factoran(X, m); %, 'rotate', 'none');

Comm = sum(lambda.^2, 2);
Uniq = 1 - Comm;
factorVars = sum(lambda.^2);

columnLabels = colLabelsFactorAnStack(1:size(lambda, 2));
rowLabels = rowLabelsFactorAnStack(1:size(lambda, 1));
matrix2latex(lambda, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% % Atttude Behavior
currSel = attitudeBehSel;
outFN = 'attitudeBeh_FactorAn.tex';

X = dataX(:, currSel);
m = 4;
[lambda,psi,T,stats,F] = factoran(X, m); %, 'rotate', 'none');

Comm = sum(lambda.^2, 2);
Uniq = 1 - Comm;
factorVars = sum(lambda.^2);

columnLabels = colLabelsFactorAnStack(1:size(lambda, 2));
rowLabels = rowLabelsFactorAnStack(1:size(lambda, 1));
matrix2latex(lambda, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% Percived Behavior Control
currSel = controlSel;
outFN = 'control_FactorAn.tex';

X = dataX(:, currSel);
m = 3;
[lambda,psi,T,stats,F] = factoran(X, m); %, 'rotate', 'none');

Comm = sum(lambda.^2, 2);
Uniq = 1 - Comm;
factorVars = sum(lambda.^2);

columnLabels = colLabelsFactorAnStack(1:size(lambda, 2));
rowLabels = rowLabelsFactorAnStack(1:size(lambda, 1));
matrix2latex(lambda, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

%% Discriminant analysis

% Atttude Cognitive
currSel = attitudeCogSel;
outFN = 'attitudeCog_Factor.tex';



X = dataX(:, currSel);
clsDrama = ClassificationDiscriminant.fit(X,yDrama);

X = dataX(:, currSel);
clsAction = ClassificationDiscriminant.fit(X,yAction);

X = dataX(:, currSel);
clsComedy = ClassificationDiscriminant.fit(X,yComedy);



% load fisheriris
% PL = meas(:,3);
% PW = meas(:,4);
% h1 = gscatter(PL,PW,species,'krb','ov^',[],'off');
% set(h1,'LineWidth',2)
% legend('Setosa','Versicolor','Virginica','Location','best')
% hold on
% X = [PL,PW];
% cls = ClassificationDiscriminant.fit(X,species);
% K = cls.Coeffs(2,3).Const; % First retrieve the coefficients for the linear
% L = cls.Coeffs(2,3).Linear;% boundary between the second and third classes
%                            % (versicolor and virginica).
% 
% % Plot the curve K + [x,y]*L  = 0.
% f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
% h2 = ezplot(f,[.9 7.1 0 2.5]);
% set(h2,'Color','r','LineWidth',2)
% 
% % Now, retrieve the coefficients for the linear boundary between the first
% % and second classes (setosa and versicolor).
% K = cls.Coeffs(1,2).Const;
% L = cls.Coeffs(1,2).Linear;
% 
% % Plot the curve K + [x1,x2]*L  = 0:
% f = @(x1,x2) K + L(1)*x1 + L(2)*x2;
% h3 = ezplot(f,[.9 7.1 0 2.5]);
% set(h3,'Color','k','LineWidth',2)
% axis([.9 7.1 0 2.5])
% xlabel('Petal Length')
% ylabel('Petal Width')
% title('{\bf Linear Classification with Fisher Training Data}')


%[beta,SIGMA,RESID,COVB,objective] = mvregress(X,Y);

%% Model: predict the relative frequency of selected genres
% model: Regress
outFormat = '%-6.1f';

% Prepare data
maxNumCoeffs = 50;
rowLabelsLogitStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsLogitStack{ii} = ['$\beta_{', num2str(ii-1), '}$'];
end

yDrama = dataX(:, gDramaQ);
yAction = dataX(:, gActionQ);
yComedy = dataX(:, gComedyQ);


columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


%% Model: predict the relative frequency of selected genres
% model: LOGIT
outFormat = '%-6.1f';

% Prepare data
maxNumCoeffs = 50;
rowLabelsLogitStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsLogitStack{ii} = ['$\beta_{', num2str(ii-1), '}$'];
end

yDrama = dataX(:, gDramaQ);
yAction = dataX(:, gActionQ);
yComedy = dataX(:, gComedyQ);


%% Model: predict the relative frequency of selected genres
% model: LDA
outFormat = '%-6.1f';

% Prepare data
maxNumCoeffs = 50;
rowLabelsLogitStack = cell(1, maxNumCoeffs);
for (ii=1:maxNumCoeffs)
    rowLabelsLogitStack{ii} = ['$\beta_{', num2str(ii-1), '}$'];
end

yDrama = dataX(:, gDramaQ);
yAction = dataX(:, gActionQ);
yComedy = dataX(:, gComedyQ);


% Atttude Cognitive
currSel = attitudeCogSel;
outFN = 'attitudeCog_Logit.tex';

X = dataX(:, currSel);
[bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');

columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Atttude Emotive
currSel = attitudeEmotSel;
outFN = 'attitudeEmot_Logit.tex';

X = dataX(:, currSel);
[bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');

columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Atttude Behavior
currSel = attitudeBehASel;
outFN = 'attitudeBehA_Logit.tex';

X = dataX(:, currSel);
[bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');

columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


% Atttude Behavior

currSel = attitudeBehBSel;
outFN = 'attitudeBehB_Logit.tex';

X = dataX(:, currSel);
[bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');

columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)

% % Subjective norms
% currSel = normsSel;
% outFN = 'normsSel_Logit.tex';
% 
% X = dataX(:, currSel);
% [bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');
% 
% X = dataX(:, currSel);
% [bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');
% 
% X = dataX(:, currSel);
% [bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');
% 
% columnLabels = rowLabelsLogitStack(1:numel(bDrama));
% rowLabels = {'Drama', 'Action', 'Comedy'};
% regCoeffs = [bDrama'; bAction'; bComedy'];
% matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
%     'alignment', 'c', 'format', outFormat)

% Percived Behavior Control
currSel = controlSel;
outFN = 'control_Logit.tex';

X = dataX(:, currSel);
[bDrama,dev,statsDrama] = glmfit(X, yDrama, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bAction,dev,statsAction] = glmfit(X, yAction, 'binomial', 'link', 'logit');

X = dataX(:, currSel);
[bComedy,dev,statsComedy] = glmfit(X, yComedy, 'binomial', 'link', 'logit');

columnLabels = rowLabelsLogitStack(1:numel(bDrama));
rowLabels = {'Drama', 'Action', 'Comedy'};
regCoeffs = [bDrama'; bAction'; bComedy'];
matrix2latex(regCoeffs, outFN, 'rowLabels', rowLabels, 'columnLabels', columnLabels, ...
    'alignment', 'c', 'format', outFormat)


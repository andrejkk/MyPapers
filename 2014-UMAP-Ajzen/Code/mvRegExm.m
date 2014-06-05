load flu;

% response: regional queries
y = double(flu(:,2:end-1));  

% predictor: national CDC estimates
x = flu.WtdILI;             
[nobs,nregions] = size(y);

% Create and fit model with separate intercepts but 
% common slope
X = cell(nobs,1);
for j=1:nobs
    X{j} = [eye(nregions), repmat(x(j),nregions,1)];
end
[beta,sig,resid,vars,loglik] = mvregress(X,y);

% Plot raw data with fitted lines
B = [beta(1:nregions)';repmat(beta(end),1,nregions)]
axes1 = axes('Position',[0.13 0.5838 0.6191 0.3412]);
xx = linspace(.5,3.5)';
h = plot(x,y,'x', xx, [ones(size(xx)),xx]*B,'-');
for j=1:nregions; 
  set(h(nregions+j),'color',get(h(j),'color')); 
end
regions = flu.Properties.VarNames;
legend1 = legend(regions{2:end-1});
set(legend1,'Position', [0.7733 0.1967 0.2161 0.6667]);

% Create and fit model with separate intercepts and slopes
for j=1:nobs
   X{j} = [eye(nregions), x(j)*eye(nregions)];
end
[beta,sig,resid,vars,loglik2] = mvregress(X,y);

% Plot raw data with fitted lines
B = [beta(1:nregions)';beta(nregions+1:end)']
axes2 = axes('Parent',gcf,'Position',...
  [0.13 0.11 0.6191 0.3412]);
h = plot(x,y,'x', xx, [ones(size(xx)),xx]*B,'-');

for j=1:nregions; 
   set(h(nregions+j),'color',get(h(j),'color')); 
end

% Likelihood ratio test for significant difference
chisq = 2*(loglik2-loglik)
p = 1-chi2cdf(chisq, nregions-1)
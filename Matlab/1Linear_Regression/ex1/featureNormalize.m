function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


% sai Comment =============checking std for whole X  whether it matches with the std after subtracting mean -- it is same actually%
fprintf("std deviateion before reducing mean");
std1=std(X(:,1));
std2=std(X(:,2));


for iter =1:size(X,2)
  mu1=mean(X(:,iter));
  
  X(:,iter)=X(:,iter).-mu1;
  mu(:,iter)=mu1;
  std1=std(X(:,iter));
  sigma(:,iter)=std1;
  X(:,iter)=X(:,iter)./std1;

end

X_norm=X
mu
sigma


% ============================================================

end

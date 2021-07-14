function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1); # n+1 becoz that 1 is for Beta not
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i


% Add ones to the X data matrix
X = [ones(m, 1) X]; # here also we defined m+1 for ones for beta not
##>> x= [1,2,3;4,5,6]
##x =
##
##   1   2   3
##   4   5   6
##
##>> m=size(x,1)
##m = 2
##>> x = [ones(m, 1) x];
##>> x
##x =
##
##   1   1   2   3
##   1   4   5   6

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


%  Set options for fmincg
initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

%  Run fmincg to obtain the optimal theta
%  This function will return theta and the cost 

%as it is onevsAll we train for each label
%labels are 1 to 10 and 10 is for number 0
fprintf("hi \n");
for c=1:num_labels
  [theta] = fmincg(@(t)(lrCostFunction(t, X, (y==c),lambda)), initial_theta, options);
  % we are using y==c becoz for first iteration y==1 will return 1 for class 1 ouput and remaining all will be 0,
  %for second iteration y==2 will return 1 for class 2 ouput and remaining all will be 0,
  all_theta(c,:)=theta'; %each row of all theta is for one classifier
end

##>> x=[1,2,3;4,5,6]
##x =
##
##   1   2   3
##   4   5   6
##
##>> x(1,:)=1
##x =
##
##   1   1   1
##   4   5   6




% =========================================================================


end

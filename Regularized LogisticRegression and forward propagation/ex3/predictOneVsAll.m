function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       
product=X*all_theta'; #First row will have predictions for first example
predictions=sigmoid(product);
%fprintf("max_Columnsare\n");

%fprintf('Program paused. Press enter to continue.\n');
%pause;

%checking whether the predictions of all classifiers in onevs all sum up to 1.

sum(predictions,2); %they are not summing to 1
%they are more or less than one as well. as shown below are some values.
%1.0206e+00
%9.9941e-01
%7.8164e-01

##>> sum(x,2)
##ans =
##
##    6
##   15
##
##>> x
##x =
##
##   1   2   3
##   4   5   6

[max_element,columnNumber]=max(predictions,[],2);
##
##>> x=[1,4,3;2,8,9]
##x =
##
##   1   4   3
##   2   8   9
##
##>> [x,y]=max(x,[],2)
##x =
##
##   4
##   9
##
##y =
##
##   2
##   3


p=columnNumber;






% =========================================================================


end

function p = predict(theta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

pred=X*theta;
pred=sigmoid(pred);
%fprintf("came here")
%pred
for iter=1:size(pred,1)
  %fprintf("came here\n")
    if pred(iter,1)>=0.5
      %fprintf("pred(iter,1) before is");
      %pred(iter,1)
      pred(iter,1)=1;
      %fprintf("pred after is")
      %pred(iter,1)
    else
      %fprintf("came to else")
      pred(iter,1)=0;
      endif

  end


% =========================================================================

p=pred;
end

function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
theta_x=X*theta;
h_of_x=zeros(size(y));
%using sigmoid instead of below three lines
%for iter=1:size(theta_x)
%  h_of_x(iter,1)=1/(1+e^(-theta_x(iter,1)));
%end

h_of_x=sigmoid(theta_x);

log_h_x=log(h_of_x); %log is applied for array of h_of_X
log_one_minus_h_x=log(1-h_of_x); %log is applied for array of (1-h_of_X is done like 1- each element of h_of_x))

product_of_y=y'*log_h_x;
product_of_one_minus_y=(1-y)'*log_one_minus_h_x;

summation=product_of_y+product_of_one_minus_y;
total=sum(product_of_y+product_of_one_minus_y);
J=(-1/m)*total;

%till here J is done %
%now work for gradients%
%for iter=1:size(theta)
diff=h_of_x-y;

grad=diff'*X;  %diff transpose * X becoz , for grad[0]= we have to multiply difference with first column of X and for grad[1] we have to multiply with 
%2nd column of x

#grad=X'*hxMinusY %above one is also correct but this seems to better correct actually.

grad=grad./m;




% =============================================================

end

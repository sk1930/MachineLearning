function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%start Computing J first
%J=(-1/m)*(Sum[{yi log h(thetaX)}+{(1-yi)*log(1-h(thetaX))}]

theta_X=X*theta ;%n*1 matrix , n number of training data
h_of_x=sigmoid(theta_X);


log_h_x=log(h_of_x); %log is applied for array of h_of_X
log_one_minus_h_x=log(1-h_of_x); %log is applied for array of (1-h_of_X is done like 1- each element of h_of_x))
yProdLogHx=y'*log_h_x; %it is a single digit

ySize=size(y,1);
onees=ones(ySize,1);

oneMinusYProdLogHx=(onees-y)'*log_one_minus_h_x; %it is a single digit
#fprintf("sumTotal is\n")
sumtotal=yProdLogHx+oneMinusYProdLogHx;
%fprintf("normal J is \n");
J=(-1/m)*sumtotal;



%till here J is done 
% now add regularization part to J
##fprintf("thetas is\n");
##
sizeoftheta=size(theta,1);
theta_except_1st=theta(2:sizeoftheta,1);
theta_except_1st_Sq=theta_except_1st.^2;
sumtheta_sq=sum(theta_except_1st_Sq);
#fprintf("regularizatonPart is \n");
regularizationPart=(lambda/(2*m))*sumtheta_sq;
J=J+regularizationPart;




%now gradient descent 
%step 1 is gradient descent without regularization
hxMinusY=h_of_x-y;
##fprintf("xis\n");
##X
##fprintf("y is\n");
##
##y
##fprintf("diff is is\n");
##
##hxMinusY
##
##fprintf("grad is  is is\n");
#grad=hxMinusY'*X #grad=X'*hxMinusY %diff transpose * X becoz , for grad[0]= we have to multiply difference with first column of X and for grad[1] we have to multiply with 
%2nd column of x

grad=X'*hxMinusY; %above one is also correct but this seems to better correct actually.
grad=grad./m;

## this is a multiline comment and to remove this select all lines ans press ctrl+R
##grad=X'*hxMinusY
##X =
##
##   1.0000   0.1000   0.6000   1.1000
##   1.0000   0.2000   0.7000   1.2000
##   1.0000   0.3000   0.8000   1.3000
##   1.0000   0.4000   0.9000   1.4000
##   1.0000   0.5000   1.0000   1.5000
##
##y is
##y =
##
##  1
##  0
##  1
##  0
##  1
##
##diff is is
##hxMinusY =
##
##  -0.3318
##   0.7109
##  -0.2497
##   0.7858
##  -0.1824
##
##grad is  is is
##grad =
##
##   0.7328
##   0.2572
##   0.6236
##   0.9900

%% if i take grad=hxMinusY'*X
%===============================
##grad=hxMinusY'*X
##
##xis
##X =
##
##   1.0000   0.1000   0.6000   1.1000
##   1.0000   0.2000   0.7000   1.2000
##   1.0000   0.3000   0.8000   1.3000
##   1.0000   0.4000   0.9000   1.4000
##   1.0000   0.5000   1.0000   1.5000
##
##y is
##y =
##
##  1
##  0
##  1
##  0
##  1
##
##diff is is
##hxMinusY =
##
##  -0.3318
##   0.7109
##  -0.2497
##   0.7858
##  -0.1824
##
##grad is  is is
##grad =
##
##   0.7328   0.2572   0.6236   0.9900



%till here gardient descent is done now add the regularization part
% it is (lambda/m)sum(theta sq)
% we add 0 to grad[1] and lambda/m * theta 2 to grad[2] ....
%fprintf("regulatization part is \n");
sizeoftheta=size(theta,1);
theta_except_1st=theta(2:sizeoftheta,1);
new_theta=[0;theta_except_1st];

regularizationPart=new_theta.*(lambda/m);


grad=grad+regularizationPart;





end

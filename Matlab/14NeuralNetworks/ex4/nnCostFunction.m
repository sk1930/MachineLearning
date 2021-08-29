function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
#feeding forward by sk
##fprintf("size of X is\n");
##size(X);   #5000*400
##fprintf("size of theta1 is\n");
##size(Theta1) #25*401
##fprintf("size of theta2 is\n");
##size(Theta2) #10*26


#add a cloumn of ones to input X
ones_for_X=size(X,1);
X=[ones(ones_for_X,1) X];
##size(X);
Hidden_layer_op=X*Theta1';
Hidden_layer_op=sigmoid(Hidden_layer_op);
##fprintf("size of hiddenlayer is \n ");
##size(Hidden_layer_op) #5000*25
ones_for_hidden=size(Hidden_layer_op,1);
Hidden_layer_op=[ones(ones_for_hidden,1) Hidden_layer_op];
#fprintf("size of hiddenlayer after adding ones \n ");

##size(Hidden_layer_op) #5000*26

output_layer=Hidden_layer_op*Theta2';
output_layer=sigmoid(output_layer);
##fprintf("size of op layers is \n");
##size(output_layer) #5000*10

#implementing Cost
##fprintf("size of y is\n");
##size(y)
Z=zeros(size(X,1),num_labels);
for i=1:size(y,1)
  Z(i,y(i,1))=1;
end
#now z is a 5000*10 matrix with each row for one example
log_h_ofx=log(output_layer);
log_one_minus_hx=log(1-output_layer);

##  x=[1,2,3;4,5,6]
##x =
##
##   1   2   3
##   4   5   6
##
##>> 1-x
##ans =
##
##   0  -1  -2
##  -3  -4  -5

#first term in cost func

##fprintf("size of loghx is\n");
##size(log_h_ofx) #5000*10
##fprintf("size of z is\n");
##size(Z)#5000*10

prod1=log_h_ofx*Z'; #taking diagonal elements in next step becoz - 
#we only multiply example 1 ouput with predicted output of example 1,
#and exmaple 2 ouput with predicyed output of exmaple 2 and so on.
prod2=log_one_minus_hx*(1-Z)';

diagona_prod1=diag(prod1);
diagona_prod2=diag(prod2);
##fprintf("size of prod1 is\n");
##size(prod1)
##fprintf("size of prod2 is\n");
##size(prod2);


total=diagona_prod1+diagona_prod2;
total;
cost1=(-1/size(X,1))*sum(total);
##fprintf("cost1 is\n");
##cost1;
J=cost1;


%start adding regularization term now
##fprintf("size of X is\n");
##size(X)   #5000*400
##fprintf("size of theta1 is\n");
##size(Theta1) #25*401
##fprintf("size of theta2 is\n");
##size(Theta2) #10*26

#we should not include the first columns of theta1 and theta2 while 
#bcoz we only sstart at theta 1 and exclude theta0

except_first_theta1=Theta1(1:size(Theta1,1),2:size(Theta1,2));
except_first_theta2=Theta2(1:size(Theta2,1),2:size(Theta2,2));
theta1Sq=except_first_theta1.^2;

theta2Sq=except_first_theta2.^2;

sum_total_regul=sum(sum(theta1Sq))+sum(sum(theta2Sq));
regu_param=[lambda/(2*size(X,1))]*sum_total_regul;
J=J+regu_param;

%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
Z=zeros(size(X,1),num_labels);
for i=1:size(y,1)
  Z(i,y(i,1))=1;
end
y=Z;
cols=size(X,2);
for i=1:size(X,1)
  #forward propagation
  
  Xi=X(i,1:cols);
  A1=Xi;#1*401
  #Theta2 size is 10*26
  #theta1 size is 25*401
  ##fprintf("error after1 \n")
  Z2=A1*Theta1';#1*25
  A2=sigmoid(Z2);
  A2=[1 A2];#1*26
  ##fprintf("error after2\n")

  Z3=A2*Theta2';#1*10
  A3=sigmoid(Z3);
  ##fprintf("sze of A3\n");
  ##size(A3) 
  #backward propagation.
  yi=y(i,1:num_labels);
  ##fprintf("sze of yi \n");
  ##size(yi) 
  ##fprintf("sze of delta3\n");
  delta3=A3-yi; #1*10 matrix
  ##size(delta3)
  delta2=delta3*Theta2; #1*26
  sigGradZ2=sigmoidGradient(Z2);
  delta2=delta2.*([1 sigGradZ2]);
  
  delta2=delta2(2:size(delta2,2));
  
  Theta1_grad=Theta1_grad+delta2'*A1;
  Theta2_grad=Theta2_grad+delta3'*A2;
  
  


end 
Theta1_grad=(1/size(X,1))*Theta1_grad;
Theta2_grad=(1/size(X,1))*Theta2_grad;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end

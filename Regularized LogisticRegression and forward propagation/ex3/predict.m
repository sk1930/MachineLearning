function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

%Note:Recall that our inputs are pixel values of
%digit images. Since the images are of size 20*20, this gives us 400 input layer
%units (excluding the extra bias unit which always outputs +1).

%The parameters have dimensions
%that are sized for a neural network with 25 units in the second layer and 10
%output units (corresponding to the 10 digit classes).

% Useful values

m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

%fprintf("x size is\n");
%size(X)
%fprintf("theta1 size is \n");
##x size is
##ans =
##
##   5000    400

##theta1 size is
##ans =
##
##    25   401
##    
##theta2 size is
##ans =
##
##   10   26

%size(Theta1)
%fprintf("paused:\n");
%pause;
%Theta1
%fprintf("theta2 size is \n");
%size(Theta2)
%fprintf("paused:\n");
%pause;
%Theta2



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

n=size(X,1);
X=[ones(n,1) X];
##
##>> x=[1,2,3;4,5,6]
##x =
##
##   1   2   3
##   4   5   6
##
##>> n=size(x,1)
##n = 2
##>> x=[zeros(n,1) x]
##x =
##
##  0    1   2   3
##  0    4   5   6
% we have 25 units each unit having 401 theta values
%

%at each layer we only apply the sigmoid function but 
%do not apply any function like >=0.5 output 1 else output 0
layer1=X*Theta1';
layer1=sigmoid(layer1);
sz=size(layer1,1);
layer1=[ones(sz,1) layer1];
layer2=layer1*Theta2';
layer2=sigmoid(layer2);
##fprintf("paused\n")
##pause;
[max_val,maxCol]=max(layer2,[],2);
p=maxCol;





% =========================================================================


end

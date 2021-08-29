function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).


g_of_z = 1.0 ./ (1.0 + exp(-z));

one_minus_g_z=1-g_of_z;
g = g_of_z.*one_minus_g_z;
#error: operator *: nonconformant arguments (op1 is 1x5, op2 is 1x5)












% =============================================================




end

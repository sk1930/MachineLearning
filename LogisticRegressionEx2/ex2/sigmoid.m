function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).



%size(z,1)
%size(z,2)
for xiter=1:size(z,1)
  %fprintf('xiter is %d',xiter)
  for yiter=1:size(z,2)
    %fprintf('yiter is %d',yiter)
    g(xiter,yiter)=(1/(1+(e^(-z(xiter,yiter)))));
  end
end
% =============================================================

end

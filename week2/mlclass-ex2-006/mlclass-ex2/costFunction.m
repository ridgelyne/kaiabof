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

size(-1.*y);
size(log(sigmoid(X*theta)));

J = 1/m * sum( (-1.*y) .* log(sigmoid(X*theta)) - (1 .- y) .* log(1.-sigmoid(X*theta))  );
%J = 1/m * sum( (-1*y') * log(sigmoid(theta
%for i = 1:size(X,2)
%	  grad(i) = 1/m * sum((sigmoid(X*theta)-y)*X(i));
%end
%grad = 1/m * sum((sigmoid(theta'*X')-y')*X);
grad = 1/m * (sigmoid(X*theta) - y)'*X;
  
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end

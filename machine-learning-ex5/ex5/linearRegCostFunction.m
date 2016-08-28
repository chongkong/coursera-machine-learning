function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

theta_ = [0; theta(2:end)];
err = X * theta - y;

unregJ = sum(err .^ 2) / (2 * m);
J = unregJ + lambda * sum(theta_ .^ 2) / (2 * m);

unregGrad = X' * err / m;
grad = unregGrad + lambda * theta_ / m;

end

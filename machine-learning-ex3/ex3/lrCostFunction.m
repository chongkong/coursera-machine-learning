function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
theta_ = [0; theta(2:end)];
h = sigmoid(X * theta);

unregJ = -sum((y .* log(h)) + ((1-y) .* log(1-h))) / m;
J = unregJ + lambda * sum(theta_' * theta_) / (2*m);

unregGrad = X' * (h - y) / m;
grad = unregGrad + lambda * theta_ / m;

end

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
m = size(X, 1);                 % number of training set
Theta1_ = Theta1(:, 2:end);     % Theta1 without bias term
Theta2_ = Theta2(:, 2:end);     % Theta2 without bias term
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% toVector(3, 5) -> [0; 0; 1; 0; 0]
% toVector(2, 10) -> [0; 1; 0; 0; 0; 0; 0; 0; 0; 0]
function vec = toVector(y, num_labels)
    vec = zeros(num_labels, 1);
    vec(y) = 1;
end

for i = 1:m
    % Forward propagation
    a1 = [1; X(i,:)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    y_ = toVector(y(i), num_labels);
    J = J - sum((y_ .* log(a3)) + ((1 - y_) .* log(1 - a3))) / m;
    
    % Backward propagation
    d3 = a3 - y_;
    d2 = (Theta2_' * d3) .* sigmoidGradient(z2); 
    
    Theta2_grad = Theta2_grad + (d3 * a2') / m;
    Theta1_grad = Theta1_grad + (d2 * a1') / m;
end

% Apply regularization term
J = J + lambda / (2 * m) * (sum(Theta1_(:) .^ 2) + sum(Theta2_(:) .^ 2));
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda * Theta1_ / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda * Theta2_ / m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

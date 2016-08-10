function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];

A2 = Theta1 * X';
A3 = Theta2 * [ones(1, m); sigmoid(A2)];
[~, p] = max(A3', [], 2);

end

function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

    function sumsqr = sumSquare(mat)
        sqrmat = mat .^ 2;
        sumsqr = sum(sqrmat(:));
    end

error = (X * Theta' - Y) .* R;
J = (sumSquare(error) + lambda * (sumSquare(X) + sumSquare(Theta))) / 2;
X_grad = error * Theta + lambda * X;
Theta_grad = error' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end

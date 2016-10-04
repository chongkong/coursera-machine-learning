function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
minCost = -1;

for C_test = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for sig_test = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sig_test));
        pred = svmPredict(model, Xval);
        cost = mean(double(pred ~= yval));
        if minCost < 0 || minCost > cost
            minCost = cost;
            C = C_test;
            sigma = sig_test;
        end
    end
end

end

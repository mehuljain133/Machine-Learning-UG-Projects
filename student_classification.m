clear; clc;

% === 1. Dataset: [Marks, Activities, Attendance, Teamwork]
X = [
    85 2 95 8;
    60 1 60 6;
    70 0 70 5;
    90 3 98 9;
    55 0 50 4;
    65 1 72 5;
    80 2 88 7;
    45 0 55 3
];

% Labels (1 = suitable, 0 = not suitable)
y = [1; 0; 0; 1; 0; 0; 1; 0];

m = size(X, 1);
input_layer_size = size(X, 2);
hidden_layer_size = 5;
num_labels = 1;

% === 2. Feature Normalization ===
mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;

% Add intercept for logistic regression
X_log = [ones(m,1), X_norm];

% === 3. Logistic Regression with Regularization ===
sigmoid = @(z) 1 ./ (1 + exp(-z));

function [J, grad] = costFunctionReg(theta, X, y, lambda)
    m = length(y);
    h = 1 ./ (1 + exp(-X * theta));
    J = (-1/m) * (y' * log(h) + (1 - y)' * log(1 - h)) + ...
        (lambda / (2 * m)) * sum(theta(2:end).^2);
    grad = (1/m) * X' * (h - y);
    grad(2:end) = grad(2:end) + (lambda / m) * theta(2:end);
end

initial_theta = zeros(size(X_log,2), 1);
lambda = 1;

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunctionReg(t, X_log, y, lambda)), initial_theta, options);

% Predict with logistic regression
predict_logistic = round(sigmoid(X_log * theta));
fprintf('Logistic Regression Accuracy: %.2f%%\n', mean(double(predict_logistic == y)) * 100);

% === 4. Neural Network with Backpropagation ===

% Sigmoid and gradient
function g = sigmoid(z)
    g = 1.0 ./ (1.0 + exp(-z));
end
function g = sigmoidGradient(z)
    g = sigmoid(z) .* (1 - sigmoid(z));
end

% Randomly initialize weights
epsilon_init = 0.12;
Theta1 = rand(hidden_layer_size, input_layer_size + 1) * 2 * epsilon_init - epsilon_init;
Theta2 = rand(num_labels, hidden_layer_size + 1) * 2 * epsilon_init - epsilon_init;

% Unroll parameters
nn_params = [Theta1(:); Theta2(:)];

% Cost function
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    m = size(X, 1);

    % Forward propagation
    X = [ones(m, 1) X];
    a2 = sigmoid([ones(m, 1) sigmoid(X * Theta1')]);
    h = sigmoid(a2 * Theta2');

    % Cost
    J = (1/m) * sum(-y .* log(h) - (1 - y) .* log(1 - h)) + ...
        lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));

    % Backpropagation
    delta3 = h - y;
    delta2 = (delta3 * Theta2)(:,2:end) .* sigmoidGradient(X * Theta1');

    Theta1_grad = (1/m) * (delta2' * X);
    Theta2_grad = (1/m) * (delta3' * a2);

    % Regularization
    Theta1_grad(:,2:end) += (lambda/m) * Theta1(:,2:end);
    Theta2_grad(:,2:end) += (lambda/m) * Theta2(:,2:end);

    grad = [Theta1_grad(:); Theta2_grad(:)];
end

% Train neural network
lambda = 1;
options = optimset('MaxIter', 500);
costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                               num_labels, X_norm, y, lambda);
[nn_params, cost] = fmincg(costFunc, nn_params, options);

% Reshape parameters
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Prediction
X_nn = [ones(m, 1) X_norm];
a2 = sigmoid([ones(m, 1) sigmoid(X_nn * Theta1')]);
pred_nn = round(sigmoid(a2 * Theta2'));

fprintf('Neural Network Accuracy: %.2f%%\n', mean(double(pred_nn == y)) * 100);

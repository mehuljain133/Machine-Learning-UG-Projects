% === 1. Training data ===
% Each row: [Marks, Activities, Attendance, Teamwork]
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

% Labels: 1 = Suitable, 0 = Not suitable
y = [1; 0; 0; 1; 0; 0; 1; 0];

m = length(y);

% === 2. Feature normalization ===
mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;

% Add intercept term
X_final = [ones(m, 1) X_norm];
n = size(X_final, 2);  % Number of features incl. bias

% === 3. Sigmoid function ===
sigmoid = @(z) 1 ./ (1 + exp(-z));

% === 4. Cost and gradient ===
function [J, grad] = costFunction(theta, X, y)
    m = length(y);
    h = 1 ./ (1 + exp(-X * theta));
    J = (-1/m) * (y' * log(h) + (1 - y)' * log(1 - h));
    grad = (1/m) * X' * (h - y);
end

% === 5. Initialize and optimize theta ===
initial_theta = zeros(n, 1);

% Use fminunc to optimize the cost function
options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(t, X_final, y)), initial_theta, options);

% === 6. Prediction Function ===
predict = @(X_input) round(sigmoid([ones(size(X_input,1), 1), (X_input - mu) ./ sigma] * theta));

% === 7. Predict a new student ===
new_student = [75 1 85 7];
prediction = predict(new_student);

fprintf('Prediction for new student: %d (1 = Suitable, 0 = Not Suitable)\n', prediction);

% === 1. Load and visualize data ===
% Sample data: Area (sq. ft) vs Price ($1000s)
area = [600; 800; 1000; 1200; 1500];  % Feature (input variable)
price = [150; 200; 250; 300; 360];    % Target (output variable)

% Plot the data
figure;
plot(area, price, 'rx', 'MarkerSize', 10);
xlabel('Area (sq ft)');
ylabel('Price ($1000s)');
title('Housing Prices vs Area');
grid on;

% === 2. Prepare data matrix X and vector y ===
m = length(price); % Number of training examples

X = [ones(m, 1), area]; % Add a column of ones to X (intercept term)
y = price;

% === 3. Compute parameters using Normal Equation ===
theta = pinv(X' * X) * X' * y;  % theta(1): intercept, theta(2): slope

fprintf('Learned parameters using Normal Equation:\n');
fprintf('Intercept (theta_0): %.2f\n', theta(1));
fprintf('Slope (theta_1): %.2f\n', theta(2));

% === 4. Make predictions ===
area_test = 1100;
X_test = [1, area_test];
predicted_price = X_test * theta;

fprintf('Predicted price for house of size %d sq ft: $%.2fK\n', area_test, predicted_price);

% === 5. Plot regression line ===
hold on;
plot(area, X * theta, '-b'); % Regression line
legend('Training data', 'Linear regression');

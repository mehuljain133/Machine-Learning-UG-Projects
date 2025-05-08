% === 1. Sample dataset ===
% Columns: [Area, Bedrooms, ServantRoom (0/1), Balconies, Age]
X = [
    1000 2 0 2 5;
    1500 3 1 3 3;
    800  2 0 1 10;
    2000 4 1 3 1;
    1200 3 0 2 7
];

% Target variable: Price in $1000s
y = [250; 400; 180; 520; 310];

m = length(y);  % Number of training examples

% === 2. Feature normalization (important for gradient descent) ===
mu = mean(X);
sigma = std(X);
X_norm = (X - mu) ./ sigma;

% Add intercept term to X
X_final = [ones(m, 1) X_norm];

% === 3. Compute theta using Normal Equation ===
theta = pinv(X_final' * X_final) * X_final' * y;

fprintf('Learned parameters (theta):\n');
disp(theta);

% === 4. Predict price for a new house ===
% New house: [Area=1600, Bedrooms=3, ServantRoom=1, Balconies=3, Age=4]
new_house = [1600 3 1 3 4];

% Normalize new input using training set mean and std
new_house_norm = (new_house - mu) ./ sigma;

% Add intercept term
new_input = [1 new_house_norm];

% Make prediction
predicted_price = new_input * theta;

fprintf('Predicted price for the new house: $%.2fK\n', predicted_price);

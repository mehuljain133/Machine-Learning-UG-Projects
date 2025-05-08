% 1. Create a sample matrix of data
A = [1 2 3; 4 5 6; 7 8 9];  % A 3x3 matrix
B = A(:);  % Convert matrix A to a column vector

% 2. Create a Histogram of the data
figure;  % Create a new figure
histogram(B, 5);  % Create a histogram with 5 bins
title('Histogram of Data from Matrix A');
xlabel('Data Values');
ylabel('Frequency');
grid on;

% 3. Plot sine and cosine functions
x = linspace(0, 2*pi, 100);  % Create 100 points from 0 to 2*pi

% Sine function
y_sin = sin(x);
% Cosine function
y_cos = cos(x);

% Create a plot with sine and cosine
figure;  % Create a new figure
plot(x, y_sin, 'r-', 'LineWidth', 2);  % Plot sine function in red
hold on;  % Hold the current plot to overlay another plot
plot(x, y_cos, 'b--', 'LineWidth', 2);  % Plot cosine function in blue (dashed)
hold off;  % Release the hold

% Add labels and legend
title('Sine and Cosine Functions');
xlabel('x values');
ylabel('y values');
legend('sin(x)', 'cos(x)', 'Location', 'best');
grid on;

% 4. Plot a scatter plot from matrix data
x_data = A(1, :);  % First row of matrix A
y_data = A(2, :);  % Second row of matrix A

figure;  % Create a new figure
scatter(x_data, y_data, 'filled', 'r');  % Scatter plot with red markers
title('Scatter Plot of Data from Matrix A');
xlabel('x (first row of A)');
ylabel('y (second row of A)');
grid on;

% 5. Plot a 3D plot from matrix data
% Assume that the matrix A is the Z values for a mesh grid
[X, Y] = meshgrid(1:size(A, 2), 1:size(A, 1));  % Create a mesh grid
Z = A;  % Z values from matrix A

figure;  % Create a new figure
surf(X, Y, Z);  % 3D surface plot
title('3D Surface Plot from Matrix A');
xlabel('X (Column indices)');
ylabel('Y (Row indices)');
zlabel('Z (Matrix values)');
colorbar;  % Add a colorbar to indicate value mapping
grid on;

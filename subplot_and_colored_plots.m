% Create some sample data
x = linspace(0, 2*pi, 100);  % x values from 0 to 2*pi
y_sin = sin(x);  % y values for sine
y_cos = cos(x);  % y values for cosine
y_random = rand(1, 10);  % 10 random values for bar plot
x_random = 1:10;  % x values for random data

% 1. Create a new figure
figure;

% 2. First subplot: Plot sine function in the first subplot (2x2 grid)
subplot(2, 2, 1);  % Create a 2x2 grid, and this is the 1st subplot
plot(x, y_sin, 'r-', 'LineWidth', 2);  % Plot sine function in red
title('Sine Function');
xlabel('x');
ylabel('sin(x)');
grid on;

% 3. Second subplot: Plot cosine function in the second subplot (2x2 grid)
subplot(2, 2, 2);  % Create the 2nd subplot in the grid
plot(x, y_cos, 'b--', 'LineWidth', 2);  % Plot cosine function in blue (dashed)
title('Cosine Function');
xlabel('x');
ylabel('cos(x)');
grid on;

% 4. Third subplot: Create a bar plot in the third subplot (2x2 grid)
subplot(2, 2, 3);  % Create the 3rd subplot in the grid
bar(x_random, y_random, 'FaceColor', 'g');  % Bar plot with green bars
title('Random Bar Plot');
xlabel('X values');
ylabel('Random Values');
grid on;

% 5. Fourth subplot: Scatter plot in the fourth subplot (2x2 grid)
subplot(2, 2, 4);  % Create the 4th subplot in the grid
scatter(x_random, y_random, 100, 'filled', 'r');  % Scatter plot with red markers
title('Scatter Plot');
xlabel('X values');
ylabel('Random Values');
grid on;

% 6. Color the plots based on a condition (e.g., change color based on y-values)
% Example for coloring sine plot based on y values (using a colormap)
figure;  % Create a new figure for the colored plot
scatter(x, y_sin, 50, y_sin, 'filled');  % Scatter plot with colored data points
colormap(jet);  % Use the 'jet' colormap (you can also try 'hot', 'cool', etc.)
colorbar;  % Display the color bar
title('Colored Sine Function');
xlabel('x');
ylabel('sin(x)');
grid on;

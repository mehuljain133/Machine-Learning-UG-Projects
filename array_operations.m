% 1. Create a single-dimensional (1D) array (vector)
single_dim_array = [1, 2, 3, 4, 5];  % Row vector
disp('Single-Dimensional Array (Row Vector):');
disp(single_dim_array);

% 2. Create a multi-dimensional (2D) array (matrix)
multi_dim_array = [1 2 3; 4 5 6; 7 8 9];  % 3x3 matrix
disp('Multi-Dimensional Array (2D Matrix):');
disp(multi_dim_array);

% 3. Create an array of all ones (3x3 matrix)
ones_array = ones(3, 3);  % 3x3 matrix of ones
disp('Array of All Ones (3x3):');
disp(ones_array);

% 4. Create an array of all zeros (3x3 matrix)
zeros_array = zeros(3, 3);  % 3x3 matrix of zeros
disp('Array of All Zeros (3x3):');
disp(zeros_array);

% 5. Create an array with random values within a specified range (e.g., 1 to 10)
random_array = randi([1, 10], 3, 3);  % 3x3 matrix with random integers between 1 and 10
disp('Array of Random Integers Between 1 and 10 (3x3):');
disp(random_array);

% 6. Create a diagonal matrix (3x3 matrix with diagonal elements)
diagonal_matrix = diag([5, 10, 15]);  % Diagonal matrix with specified values
disp('Diagonal Matrix (3x3):');
disp(diagonal_matrix);

% 7. Create a 3D array (multi-dimensional array)
three_dim_array = rand(3, 3, 3);  % 3x3x3 array with random values between 0 and 1
disp('3D Array (3x3x3) with Random Values:');
disp(three_dim_array);

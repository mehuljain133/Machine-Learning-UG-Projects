% 1. Create a sample matrix
A = [1 2 3; 4 5 6; 7 8 9];  % A 3x3 matrix

% 2. Compute the size of a matrix
matrix_size = size(A);  % Returns the size of the matrix A (rows, columns)
disp('Size of the matrix A:');
disp(matrix_size);

% 3. Compute the size of a particular row or column
row_size = size(A, 1);   % Number of rows in matrix A
col_size = size(A, 2);   % Number of columns in matrix A
disp('Number of rows in A:');
disp(row_size);
disp('Number of columns in A:');
disp(col_size);

% Length of a particular row (say row 2)
row_2_length = length(A(2, :));  % Length of row 2
disp('Length of row 2 in matrix A:');
disp(row_2_length);

% Length of a particular column (say column 3)
col_3_length = length(A(:, 3));  % Length of column 3
disp('Length of column 3 in matrix A:');
disp(col_3_length);

% 4. Load data from a text file
% Ensure you have a file called 'data.txt' with numbers in the current directory
% Example content of data.txt:
% 1 2 3
% 4 5 6
% 7 8 9
% Load the data from the text file into matrix 'B'
B = load('data.txt');  
disp('Loaded data from text file:');
disp(B);

% 5. Store matrix data to a text file
% Save matrix A to a file called 'output.txt'
save('output.txt', 'A', '-ascii');
disp('Matrix A has been saved to "output.txt".');

% 6. Find out variables and their features in the current scope
whos;  % Lists all variables in the current workspace along with their sizes and types

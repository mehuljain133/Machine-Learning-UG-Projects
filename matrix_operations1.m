% 1. Create two sample matrices
A = [1 2 3; 4 5 6; 7 8 9];  % 3x3 matrix
B = [9 8 7; 6 5 4; 3 2 1];  % Another 3x3 matrix

disp('Matrix A:');
disp(A);

disp('Matrix B:');
disp(B);

% 2. Matrix Addition (A + B)
addition_result = A + B;
disp('Matrix Addition (A + B):');
disp(addition_result);

% 3. Matrix Subtraction (A - B)
subtraction_result = A - B;
disp('Matrix Subtraction (A - B):');
disp(subtraction_result);

% 4. Matrix Multiplication (A * B)
multiplication_result = A * B;  % Matrix multiplication
disp('Matrix Multiplication (A * B):');
disp(multiplication_result);

% 5. Display specific rows or columns of the matrix

% Display row 2 of matrix A
row_2 = A(2, :);  % Get the 2nd row of matrix A
disp('Row 2 of Matrix A:');
disp(row_2);

% Display column 3 of matrix A
col_3 = A(:, 3);  % Get the 3rd column of matrix A
disp('Column 3 of Matrix A:');
disp(col_3);

% Display specific element (row 1, column 2)
element_1_2 = A(1, 2);  % Get the element at row 1, column 2
disp('Element at row 1, column 2 of Matrix A:');
disp(element_1_2);

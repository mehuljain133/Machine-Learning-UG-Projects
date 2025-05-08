% Define two sample matrices
A = [1 2 3; 4 5 6; 7 8 9];  % 3x3 matrix
B = [9 8 7; 6 5 4; 3 2 1];  % 3x3 matrix

% 1. Transpose of a matrix
A_transpose = A';  % Use the apostrophe (') operator to transpose
disp('Transpose of Matrix A:');
disp(A_transpose);

% 2. Addition of two matrices
C_add = A + B;  % Element-wise addition
disp('Addition of A and B:');
disp(C_add);

% 3. Subtraction of two matrices
C_sub = A - B;  % Element-wise subtraction
disp('Subtraction of B from A:');
disp(C_sub);

% 4. Element-wise multiplication
C_elem_mult = A .* B;  % Use .* for element-wise multiplication
disp('Element-wise multiplication of A and B:');
disp(C_elem_mult);

% 5. Matrix multiplication
C_mat_mult = A * B;  % Use * for matrix multiplication
disp('Matrix multiplication of A and B:');
disp(C_mat_mult);

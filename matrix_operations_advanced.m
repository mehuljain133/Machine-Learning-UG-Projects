% 1. Create a sample matrix
A = [-1 2 -3; 4 -5 6; -7 8 -9];  % 3x3 matrix with both positive and negative values

disp('Matrix A:');
disp(A);

% 2. Convert matrix data to absolute values
abs_A = abs(A);  % Absolute values of matrix A
disp('Absolute values of Matrix A:');
disp(abs_A);

% 3. Take the negative of matrix values
neg_A = -A;  % Negate all elements of A
disp('Negative values of Matrix A:');
disp(neg_A);

% 4. Add a row to the matrix (append row at the end)
new_row = [10 11 12];  % New row to add
A_with_new_row = [A; new_row];  % Add new row at the bottom
disp('Matrix A with new row added:');
disp(A_with_new_row);

% 5. Add a column to the matrix (append column at the end)
new_column = [13; 14; 15; 16];  % New column to add
A_with_new_column = [A_with_new_row, new_column];  % Add new column
disp('Matrix A with new column added:');
disp(A_with_new_column);

% 6. Remove a row from the matrix (remove the 2nd row)
A_without_row = A(1:end ~= 2, :);  % Remove the 2nd row (keep all other rows)
disp('Matrix A with the 2nd row removed:');
disp(A_without_row);

% 7. Remove a column from the matrix (remove the 3rd column)
A_without_column = A(:, 1:end ~= 3);  % Remove the 3rd column (keep all other columns)
disp('Matrix A with the 3rd column removed:');
disp(A_without_column);

% 8. Find the maximum value in the entire matrix
max_value = max(A(:));  % Maximum value in matrix A
disp('Maximum value in Matrix A:');
disp(max_value);

% 9. Find the minimum value in the entire matrix
min_value = min(A(:));  % Minimum value in matrix A
disp('Minimum value in Matrix A:');
disp(min_value);

% 10. Find the maximum value in each column
max_in_columns = max(A);  % Maximum value in each column
disp('Maximum values in each column of Matrix A:');
disp(max_in_columns);

% 11. Find the minimum value in each row
min_in_rows = min(A, [], 2);  % Minimum value in each row
disp('Minimum values in each row of Matrix A:');
disp(min_in_rows);

% 12. Find the sum of all elements in the matrix
sum_all_elements = sum(A(:));  % Sum of all elements in matrix A
disp('Sum of all elements in Matrix A:');
disp(sum_all_elements);

% 13. Find the sum of elements in each column
sum_in_columns = sum(A);  % Sum of elements in each column
disp('Sum of elements in each column of Matrix A:');
disp(sum_in_columns);

% 14. Find the sum of elements in each row
sum_in_rows = sum(A, 2);  % Sum of elements in each row
disp('Sum of elements in each row of Matrix A:');
disp(sum_in_rows);

% 1. Conditional Statements (if-else)

% Define a variable
x = 5;

% Check if the number is positive, negative, or zero
if x > 0
    disp('x is positive.');
elseif x < 0
    disp('x is negative.');
else
    disp('x is zero.');
end

% 2. for Loop: Summing the first N natural numbers

N = 10;  % Define the number of terms
sum_result = 0;  % Initialize the sum

% Using a for loop to calculate the sum of the first N natural numbers
for i = 1:N
    sum_result = sum_result + i;  % Add the current number to the sum
end

disp(['The sum of the first ', num2str(N), ' natural numbers is: ', num2str(sum_result)]);

% 3. while Loop: Print numbers from 1 to 5 using a while loop

i = 1;  % Initialize the counter

% Using a while loop to print numbers from 1 to 5
while i <= 5
    disp(['Number: ', num2str(i)]);
    i = i + 1;  % Increment the counter
end

% 4. Nested for Loop: Printing a multiplication table

num = 3;  % Define the number to create the multiplication table

% Using a nested for loop to print the multiplication table of the number
disp(['Multiplication table for ', num2str(num), ':']);
for i = 1:10
    for j = 1:10
        product = i * j;  % Multiply i and j
        fprintf('%d x %d = %d\n', i, j, product);  % Display the result
    end
end

% 5. Continue and Break in a Loop: Example to break and continue based on a condition

% Loop through numbers from 1 to 10
for i = 1:10
    if i == 3
        continue;  % Skip the iteration when i is 3
    elseif i == 8
        break;  % Exit the loop when i is 8
    end
    disp(['Number: ', num2str(i)]);
end

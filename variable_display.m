% 1. Create and initialize numerical variables
num1 = 10;              % An integer
num2 = 3.14159;         % A floating point number

% 2. Display numerical variables using simple formatting
fprintf('The value of num1 (integer): %d\n', num1);
fprintf('The value of num2 (float): %.2f\n', num2); % Show 2 decimal places

% 3. Create and initialize string variables
str1 = 'Hello, World!'; % A simple string
str2 = 'MATLAB is fun.'; % Another simple string

% 4. Display strings
disp('Displaying strings:');
disp(str1);
disp(str2);

% 5. Concatenate strings and display with formatting
greeting = strcat(str1, ' ', str2); % Concatenate strings
fprintf('Concatenated Greeting: %s\n', greeting);

% 6. Use mixed formatting (both strings and numbers)
name = 'Alice';
age = 25;
fprintf('%s is %d years old.\n', name, age);

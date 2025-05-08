% Define two logical variables (true or false)
x = true;
y = false;

% 1. Logical OR (x OR y)
or_result = x || y;
fprintf('Logical OR (x OR y): %d || %d = %d\n', x, y, or_result);

% 2. Logical AND (x AND y)
and_result = x && y;
fprintf('Logical AND (x AND y): %d && %d = %d\n', x, y, and_result);

% 3. Checking for Equality (x == y)
equality_result = (x == y);
fprintf('Equality check (x == y): %d == %d = %d\n', x, y, equality_result);

% 4. Logical NOT (NOT x)
not_result = ~x;  % Logical NOT
fprintf('Logical NOT (NOT x): ~%d = %d\n', x, not_result);

% 5. Logical XOR (x XOR y)
xor_result = xor(x, y);
fprintf('Logical XOR (x XOR y): %d XOR %d = %d\n', x, y, xor_result);

# Regression: Linear regression with one variable, linear regression with multiple variables,gradient descent, logistic regression, over-fitting, regularization. performance evaluation metrics, validation methods. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.datasets import make_classification

# Function to implement Gradient Descent for Linear Regression
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        predictions = X.dot(theta)
        errors = predictions - y
        theta -= (learning_rate / m) * X.T.dot(errors)  # Update rule
        cost = (1 / (2 * m)) * np.sum(errors ** 2)
        cost_history.append(cost)
    return theta, cost_history

# Generate a simple dataset for linear regression (single variable)
X = np.random.rand(100, 1) * 10  # Feature
y = 2 * X + 3 + np.random.randn(100, 1) * 2  # Target with some noise

# Add a column of ones to X to account for the intercept term in linear regression
X_b = np.c_[np.ones((len(X), 1)), X]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)

# Linear Regression with one variable
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred = lin_reg.predict(X_test)

# Performance evaluation (for linear regression)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression (One variable) - RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Visualizing the result
plt.scatter(X_test[:, 1], y_test, color='blue', label='True values')
plt.plot(X_test[:, 1], y_pred, color='red', label='Prediction')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression (One Variable)')
plt.legend()
plt.show()

# Now for Linear Regression with Multiple Variables
# Creating a dataset with multiple features
X_multi = np.random.rand(100, 3) * 10  # 3 features
y_multi = 3 * X_multi[:, 0] + 2 * X_multi[:, 1] + 4 * X_multi[:, 2] + 5 + np.random.randn(100, 1) * 2

# Add intercept term
X_multi_b = np.c_[np.ones((len(X_multi), 1)), X_multi]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_multi_b, y_multi, test_size=0.2, random_state=42)

# Linear Regression with multiple variables
lin_reg_multi = LinearRegression()
lin_reg_multi.fit(X_train, y_train)

# Predictions
y_pred_multi = lin_reg_multi.predict(X_test)

# Performance evaluation (for multiple variables linear regression)
mse_multi = mean_squared_error(y_test, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_test, y_pred_multi)

print(f"Linear Regression (Multiple variables) - RMSE: {rmse_multi:.2f}, R²: {r2_multi:.2f}")

# Gradient Descent for Linear Regression
theta_initial = np.random.randn(2, 1)  # Random initialization of theta (2 parameters: intercept + coefficient)
learning_rate = 0.01
iterations = 1000

theta, cost_history = gradient_descent(X_train, y_train, theta_initial, learning_rate, iterations)

print(f"Gradient Descent - Final Theta: {theta.ravel()}")

# Logistic Regression (for classification task)
# Generating a synthetic binary classification dataset
X_class, y_class = make_classification(n_samples=100, n_features=5, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_class, y_class)

# Predictions
y_pred_class = log_reg.predict(X_class)

# Performance evaluation (for logistic regression)
accuracy = accuracy_score(y_class, y_pred_class)
cm = confusion_matrix(y_class, y_pred_class)

print(f"Logistic Regression - Accuracy: {accuracy:.2f}")
print("Confusion Matrix:\n", cm)

# Regularization in Linear Regression (Ridge and Lasso)
from sklearn.linear_model import Ridge, Lasso

# Ridge Regression (L2 Regularization)
ridge_reg = Ridge(alpha=1.0)
ridge_reg.fit(X_train[:, 1:], y_train)
ridge_pred = ridge_reg.predict(X_test[:, 1:])
ridge_mse = mean_squared_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"Ridge Regression - RMSE: {ridge_rmse:.2f}, R²: {ridge_r2:.2f}")

# Lasso Regression (L1 Regularization)
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train[:, 1:], y_train)
lasso_pred = lasso_reg.predict(X_test[:, 1:])
lasso_mse = mean_squared_error(y_test, lasso_pred)
lasso_rmse = np.sqrt(lasso_mse)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"Lasso Regression - RMSE: {lasso_rmse:.2f}, R²: {lasso_r2:.2f}")

# Cross-validation for model validation
cv_scores = cross_val_score(lin_reg_multi, X_multi_b, y_multi, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-validation MSE for Linear Regression with multiple variables: {-cv_scores.mean():.2f}")

# Overfitting example (too complex a model on a small dataset)
# We simulate overfitting by using a high-degree polynomial on a small dataset.
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_reg = make_pipeline(PolynomialFeatures(degree=15), LinearRegression())
poly_reg.fit(X_train[:, 1].reshape(-1, 1), y_train)

# Predictions on the same training set (overfitting)
y_train_pred_poly = poly_reg.predict(X_train[:, 1].reshape(-1, 1))

# Performance on the training set
train_rmse_poly = np.sqrt(mean_squared_error(y_train, y_train_pred_poly))
print(f"Polynomial Regression (degree 15) - Overfitting RMSE: {train_rmse_poly:.2f}")

# Introduction: Basic definitions, Hypothesis space and inductive bias, Bayes optimal classifierand Bayes error, Occam's razor, Curse of dimensionality, dimensionality reduction, feature scaling, feature selection methods. 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# Load a dataset (for simplicity, we'll use the Iris dataset)
data = load_iris()
X = data.data
y = data.target

# Basic Definitions
# Hypothesis Space: All possible models (classifiers) that can be learned from the data.
# Inductive Bias: Assumptions made to generalize beyond the training data, like using a random forest.
# Bayes Optimal Classifier: The classifier that minimizes the Bayes error, based on the true distribution.

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling: Standardization to make the model invariant to the scale of the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dimensionality Reduction using PCA (Principal Component Analysis)
# To mitigate the curse of dimensionality and improve model performance
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Visualizing data after PCA (Dimensionality Reduction)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_train_pca[:, 0], y=X_train_pca[:, 1], hue=y_train, palette='Set1')
plt.title('Data after PCA')
plt.show()

# Feature Selection Methods (Example: Random Forest feature importance)
# Train a Random Forest classifier to select important features
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
importances = rf_classifier.feature_importances_

# Plot feature importances
plt.bar(range(len(importances)), importances)
plt.title('Feature Importances from Random Forest')
plt.xlabel('Feature Index')
plt.ylabel('Importance')
plt.show()

# Now let's train a classifier (Random Forest) to make predictions
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix for performance evaluation
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix')
plt.show()

# Occam's Razor: Simpler models (lower complexity) are preferred if they perform similarly to more complex models.
# In our case, Random Forest is relatively complex, but you can compare it with simpler classifiers like Logistic Regression.

# Curves: Adding the concept of dimensionality reduction using PCA
print(f"Explained Variance by First Two Principal Components: {np.sum(pca.explained_variance_ratio_)}")

# Bayes Error is typically the minimum error achievable based on the distribution of data and its noise. 
# A classifier can only get so close to this, depending on how well it models the true underlying distribution.


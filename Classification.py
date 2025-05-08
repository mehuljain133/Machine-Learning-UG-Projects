# Classification: Decision trees, Naive Bayes classifier, k-nearest neighbor classifier, perceptron,multilayer perceptron, neural networks, back-propagation algorithm, Support Vector Machine (SVM), Kernel functions. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
import seaborn as sns

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling for some classifiers like k-NN, SVM, and Perceptron
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
y_pred_dt = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

# 2. Naive Bayes Classifier (GaussianNB)
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
y_pred_nb = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")

# 3. k-Nearest Neighbors (k-NN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)
y_pred_knn = knn_classifier.predict(X_test_scaled)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
print(f"k-NN Accuracy: {knn_accuracy:.2f}")

# 4. Perceptron (Single-layer Neural Network)
perceptron_classifier = Perceptron(random_state=42)
perceptron_classifier.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron_classifier.predict(X_test_scaled)
perceptron_accuracy = accuracy_score(y_test, y_pred_perceptron)
print(f"Perceptron Accuracy: {perceptron_accuracy:.2f}")

# 5. Multilayer Perceptron (MLP) - Neural Network
mlp_classifier = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp_classifier.fit(X_train_scaled, y_train)
y_pred_mlp = mlp_classifier.predict(X_test_scaled)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLP Accuracy: {mlp_accuracy:.2f}")

# 6. Support Vector Machine (SVM) with linear kernel
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)
y_pred_svm = svm_classifier.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM (Linear Kernel) Accuracy: {svm_accuracy:.2f}")

# 7. Support Vector Machine (SVM) with RBF kernel
svm_rbf_classifier = SVC(kernel='rbf', random_state=42)
svm_rbf_classifier.fit(X_train_scaled, y_train)
y_pred_svm_rbf = svm_rbf_classifier.predict(X_test_scaled)
svm_rbf_accuracy = accuracy_score(y_test, y_pred_svm_rbf)
print(f"SVM (RBF Kernel) Accuracy: {svm_rbf_accuracy:.2f}")

# Confusion Matrix for the best performing model (for example, MLP)
cm = confusion_matrix(y_test, y_pred_mlp)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Confusion Matrix for MLP')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Visualizing decision boundaries for SVM (RBF kernel) as an example
from sklearn.decomposition import PCA

# Reduce to 2D using PCA for visualization
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

svm_rbf_classifier_2d = SVC(kernel='rbf', random_state=42)
svm_rbf_classifier_2d.fit(X_train_2d, y_train)

# Plot decision boundaries
h = .02  # step size in the mesh
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = svm_rbf_classifier_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o', s=50)
plt.title('SVM (RBF Kernel) Decision Boundaries')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()


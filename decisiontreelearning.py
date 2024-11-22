# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset into a DataFrame
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

# EDA - Exploratory Data Analysis
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset Summary:")
print(data.describe())

print("\nTarget Class Distribution:")
print(data['target'].value_counts())

# Visualize class distribution
sns.countplot(x='target', data=data)
plt.title('Class Distribution')
plt.show()

# Pairplot to visualize feature relationships
sns.pairplot(data, hue='target', diag_kind='kde', markers=['o', 's', 'D'])
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix")
plt.show()

# Prepare the data for Decision Tree (CART)
X = data.drop(columns=['target'])
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree using CART
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(report)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree")
plt.show()

# Perform GridSearchCV for Pruning
param_grid = {
    'max_depth': [None, 2, 3, 4],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3]
}
grid_search = GridSearchCV(DecisionTreeClassifier(criterion='gini', random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters after pruning
best_params = grid_search.best_params_
print("Best Parameters after Pruning:", best_params)

# Train a pruned Decision Tree
pruned_model = grid_search.best_estimator_
pruned_model.fit(X_train, y_train)

# Predict using the pruned model
y_pruned_pred = pruned_model.predict(X_test)
pruned_accuracy = accuracy_score(y_test, y_pruned_pred)
print(f"Pruned Model Accuracy: {pruned_accuracy}")

# Visualize the Pruned Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(pruned_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Pruned Decision Tree")
plt.show()

# Classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Replace with your sample features
predicted_class = pruned_model.predict(new_sample)
print(f"Predicted Class for the New Sample: {iris.target_names[int(predicted_class)]}")

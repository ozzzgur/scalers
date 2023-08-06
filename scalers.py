import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
### Important you need to adjust the limitation of X and Y by using xlim(a,b) and ylim(a,b) or if you prefer you can erase them.
# Create a synthetic dataset with outliers and skewed features
X, y = make_classification(n_samples=100000, n_features=10, n_informative=5, n_redundant=2, random_state=42)

# Introduce outliers to the first feature
np.random.seed(42)
outlier_indices = np.random.choice(X.shape[0], size=50, replace=False)
X[outlier_indices, 0] = X[outlier_indices, 0] + 10

# Introduce skewed distributions to the last two features
X[:, -2] = np.exp(X[:, -2])  # Apply the inverse of the log transformation to get valid values
X[:, -1] = X[:, -1]**2  # Square the values to introduce skewness

# Make the dataset imbalanced
np.random.seed(42)
class1_indices = np.where(y == 1)[0]
class1_indices_to_keep = np.random.choice(class1_indices, size=int(0.2 * len(class1_indices)), replace=False)
class0_indices = np.where(y == 0)[0]
indices_to_keep = np.concatenate((class1_indices_to_keep, class0_indices))
X = X[indices_to_keep]
y = y[indices_to_keep]

# Split the dataset into training and testing sets


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

results = {}

for scaler_name, scaler in scalers.items():
    # Fit and transform the training data using the scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression classifier
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test_scaled)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    results[scaler_name] = accuracy

    # Draw histogram,boxplot,scatterplot to scaled X_train_scaled
    # Convert the scaled training data back to a DataFrame for visualization
    df_train_scaled = pd.DataFrame(X_train_scaled, columns=[f"feature_{i}" for i in range(1, 11)])

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.title(f"Histogram of Features in {scaler_name} DataFrame")
    plt.hist(df_train_scaled.values.flatten(), bins=1000, alpha=0.6)
    plt.xlabel("Feature Values")
    plt.xlim(-0.15,1)
    plt.ylim(0,6000)

    plt.ylabel("Frequency")
    plt.grid(False)
    # Plot histogram of features in one picture
    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Histogram of Features in {scaler_name}", fontsize=16)

    for i, col in enumerate(df_train_scaled.columns, 1):
        plt.subplot(3, 4, i)
        plt.hist(df_train_scaled[col], bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        plt.title(f"{col}", fontsize=12)
        plt.grid(axis='y')

    plt.tight_layout()
    plt.show(block=True)

    # Plot boxplot
    plt.figure(figsize=(12, 6))
    plt.title(f"Boxplot of Features - {scaler_name}")
    sns.boxplot(data=df_train_scaled, palette="Set3",showmeans=True)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()


    # Plot scatter plot
    plt.figure(figsize=(8, 6))
    plt.title(f"Scatter Plot of Features - {scaler_name}")
    sns.scatterplot(data=df_train_scaled, x="feature_1", y="feature_2", hue=y_train, palette="Set1", alpha=0.5)
    plt.tight_layout()
plt.show()
# Sort the results by accuracy in descending order
sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}

# Print the results
for rank, (scaler_name, accuracy) in enumerate(sorted_results.items(), 1):
    print(f"{rank}. {scaler_name}: {accuracy:.4f}")




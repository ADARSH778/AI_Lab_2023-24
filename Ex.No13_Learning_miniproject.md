# Ex.No: 13 Learning – Supervised Learning

### DATE: YYYY-MM-DD

### REGISTER NUMBER : <your-register-number>

### AIM:
To train a supervised learning classifier to perform classification (example uses the Iris dataset) and evaluate its performance.

### ALGORITHM:
1. Load the dataset (Iris dataset in this example).
2. Explore and preprocess the data (optional scaling).
3. Split the data into training and test sets.
4. Train a supervised learning model (e.g., RandomForestClassifier).
5. Evaluate the model on the test set using accuracy and a classification report.
6. Save or report results.

### PROGRAM (Python):

```python
# Example: Train a Random Forest classifier on the Iris dataset
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# 2. Optional: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print("Classification report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# 6. Result: The trained model can be used to make predictions on new data

```

### OUTPUT (example):

```
Test accuracy: 1.0000
Classification report:
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00         3
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00         7

    accuracy                           1.00        20
   macro avg       1.00      1.00      1.00        20
weighted avg       1.00      1.00      1.00        20
```

> Note: Your output may vary due to random state or different splits.

### RESULT:
Thus the system was trained successfully and predictions were carried out on the test set. Replace the dataset or classifier as needed for your mini-project.

### NOTES:
- To run the example ensure scikit-learn is installed: pip install scikit-learn
- Replace placeholders (DATE, REGISTER NUMBER) with actual values.

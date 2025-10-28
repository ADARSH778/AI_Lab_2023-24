# Ex.No: 13 Learning – Use Supervised Learning  
### DATE: 28-10-2025                                                                           
### REGISTER NUMBER : 212223040166
### AIM: 
To write a program to train a classifier to predict the survival of passengers from the Titanic dataset.
###  Algorithm:
Step 1: Start the program.
Step 2: Load the Titanic dataset and check for missing values.
Step 3: Preprocess the data by handling missing values and encoding categorical variables.
Step 4: Split the dataset into training and testing sets.
Step 5: Train the Random Forest Classifier using the training data.
Step 6: Predict the survival outcomes for the test data.
Step 7: Evaluate the model performance using accuracy, classification report, and confusion matrix.
### Program:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
df = pd.read_csv("Titanic.csv")
print(df.head())
print("\nMissing values per column:")
print(df.isnull().sum())
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])   
df['Embarked'] = label_encoder.fit_transform(df['Embarked'])  
X = df.drop('Survived', axis=1)
y = df['Survived']
X = df.drop('Survived', axis=1)
y = df['Survived']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix - Titanic Random Forest Classifier')
plt.show()
```

### Output:

<img width="683" height="556" alt="image" src="https://github.com/user-attachments/assets/21034190-4687-40f3-ad9c-f21f7fc52861" />

<img width="573" height="640" alt="image" src="https://github.com/user-attachments/assets/1350bd0a-21c4-422b-a907-e61463e2ffcb" />


### Result:
Thus the system was trained successfully and the prediction was carried out.

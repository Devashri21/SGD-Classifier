# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries (pandas, numpy, sklearn).

2. Load the Iris dataset from sklearn.datasets.

3. Split the dataset into training and testing sets.

4. Initialize the SGDClassifier with loss="log_loss" (for logistic regression).

5. Train the model using the training dataset.

6. Predict the species for the test dataset.

7. Evaluate model performance using accuracy score and classification report.

## Program:
```

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy of the model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


```

## Output:

<img width="722" height="298" alt="Screenshot 2025-10-07 103958" src="https://github.com/user-attachments/assets/30bfae09-370a-485c-866a-22f61c2610ab" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.

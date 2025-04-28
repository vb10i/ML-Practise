import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(r'C:\Users\vansh\Desktop\ml\creditcard\creditcard.csv')
df = df.drop(columns=['Time'])

X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(max_depth=3, criterion='entropy', class_weight={0:1, 1:20}, min_samples_leaf=3, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)

print(acc)
cofmat = confusion_matrix(y_test, y_predict)
print(cofmat)

roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(roc_auc)


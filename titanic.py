import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv(r'C:\Users\vansh\Desktop\ml\titanic\train.csv')
df_test = pd.read_csv(r'C:\Users\vansh\Desktop\ml\titanic\test.csv')
df_test_labels = pd.read_csv(r'C:\Users\vansh\Desktop\ml\titanic\gender_submission.csv')
df_test = df_test.merge(df_test_labels, on="PassengerId")

df_train = df_train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].mode()[0])
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())

encoder = LabelEncoder()
for col in ['Sex', 'Embarked']:
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

X = df_train.drop(columns=['Survived'])
y = df_train['Survived']

X_test = df_test.drop(columns=['Survived'])
y_test = df_test['Survived']

model1 = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=42)
model1.fit(X, y)

y_predict = model1.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print(acc)

confusionmatrix = confusion_matrix(y_test, y_predict)
print(confusionmatrix)

rocauc = roc_auc_score(y_test, model1.predict_proba(X_test)[:, 1])
print(rocauc)

model_rf = RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy', random_state=42)
model_rf.fit(X, y)

y_predictt = model_rf.predict(X_test)

accc = accuracy_score(y_test, y_predictt)
print(accc)

confusionmatrixx = confusion_matrix(y_test, y_predictt)
print(confusionmatrixx)

rocaucc = roc_auc_score(y_test, model_rf.predict_proba(X_test)[:, 1])
print(rocaucc)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)

print(df.shape)
print(df.head)

X = df.drop('Fitness', axis=1)
y = df['Fitness']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(y_pred)

print(classification_report(y_test, y_pred))
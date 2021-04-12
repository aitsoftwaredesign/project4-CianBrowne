import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)

print(df.shape)
print(df.head)

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print(y_pred)

print(classification_report(y_test, y_pred))
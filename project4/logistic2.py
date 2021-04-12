import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)

X = df.drop('Fitness', axis=1)
y = df['Fitness']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

clf = LogisticRegression(solver='liblinear', C=1, multi_class='auto')

clf.fit(X,y)

predictions = clf.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
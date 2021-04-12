import pandas as pd

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix
data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)

del df['region']
category = pd.cut(df['charges'],bins=[0,3000,5000,10000,25000,100000],labels=[0,1,2,3,4])
df.insert(5, 'test', category)
#print(df['test'].unique())
#print(df)
X = df.drop('test', axis=1)
y = df['test']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

clf = LogisticRegression(solver='liblinear', C=1, multi_class='auto')

clf.fit(X,y)

predictions = clf.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
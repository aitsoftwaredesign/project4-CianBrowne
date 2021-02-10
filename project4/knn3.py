import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
from sklearn.metrics import classification_report, confusion_matrix

import joblib


#Self rated level of fitness
data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)
X = df[['Miles', 'Age', 'Income', 'Gender', 'MaritalStatus', 'Education']].values
y = data[['Fitness']]
print(data)

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print('predictions: ', predictions)
print('accuracy: ', accuracy)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
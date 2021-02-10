import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import joblib


#Self rated level of fitness
data = pd.read_csv("cardio_train.csv")
df = pd.DataFrame(data)
X = data[['age', 'weight', 'height']].values
y = data[['cardio']]
print(data)

knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print('predictions: ', predictions)
print('accuracy: ', accuracy)


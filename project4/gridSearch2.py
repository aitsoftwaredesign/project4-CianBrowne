from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import time
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 600)

data = pd.read_csv("cardio_train.csv")
df = pd.DataFrame(data[1:1000])


X = df.drop('cardio', axis=1)
y = df['cardio']
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
clf = GridSearchCV(SVC(), {
    'C': [1,3,5,10,20],
    'kernel': ['rbf','linear', 'poly', 'sigmoid']
}, cv=5, return_train_score=False, n_jobs=-1)
clf.fit(X, y)
df2 = pd.DataFrame(clf.cv_results_)
print(df2)
print(clf.best_params_)
print(clf.best_score_)
print(time.time() - start_time, 's')





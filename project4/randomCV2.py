from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import progressbar
import time
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 600)

data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)

del df['region']

#print(df.shape)
#print(df.head)
category = pd.cut(df['charges'],bins=[0,2000,5000,10000,25000,50000,100000],labels=[0,1,2,3,4,5])
df.insert(5, 'test', category)
#print(df['test'].unique())
#print(df)
X = df.drop('test', axis=1)
y = df['test']


print(y)
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
clf = GridSearchCV(SVC(), {
    'C': [1,3,5,10,20],
    'kernel': ['rbf','linear', 'poly', 'sigmoid']
    #'kernel': [ 'linear' ]
}, cv=5, return_train_score=False, n_jobs=-1)
clf.fit(X, y)
df2 = pd.DataFrame(clf.cv_results_)
print(df2)
print(clf.best_params_)
print(clf.best_score_)
print(time.time() - start_time, 's')




# rs = RandomizedSearchCV(SVC(gamma='auto'){
#     'C': [1,5,10,20,30,50,100],
#     'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
#
#     },
#     cv=35,
#     return_train_score=False,
#     n_iter=2
#
#
# )
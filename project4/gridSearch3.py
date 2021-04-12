from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
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
start_time = time.time()

data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)

#print(df.shape)
#print(df.head)

X = df.drop('Fitness', axis=1)
y = df['Fitness']
print(y)
clf = GridSearchCV(SVC(), {
    'C': [1,3,5,10],
    'kernel': ['rbf', 'poly', 'sigmoid']
    #'kernel': [ 'linear' ]
}, cv=5, return_train_score=False, n_jobs=-1)
clf.fit(X, y)
df2 = pd.DataFrame(clf.cv_results_)
print(df2)
print(clf.best_params_)
print(clf.best_score_)
print(time.time() - start_time, 's')
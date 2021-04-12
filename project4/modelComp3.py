from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
import sklearn.cluster as cluster
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

model_params = {
    'svm': {
        'model': SVC(),
        'params': {
            'C': [1,2,3,5,10],
            'kernel': ['rbf','linear', 'poly', 'sigmoid']
        }
    },
    'random_forest':{
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [1,5,10]
        }
    },
'logistic_regression' : {
        'model': LogisticRegression(),
        'params': {
            'solver': ['liblinear'],
            'C': [1,5,10],
            'multi_class': ['auto']
        }
    },
    'linear_regression': {
        'model': LinearRegression(),
        'params':{
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(),
        'params':{
            'splitter': ['best', 'random']
        }
    },
    'KNN': {
        'model': neighbors.KNeighborsClassifier(),
        'params':{
            'n_neighbors': [2,3,5,10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }

}

scores = []

#print(df.shape)
#print(df.head)
del df['region']
category = pd.cut(df['charges'],bins=[0,3000,5000,10000,25000,100000],labels=[0,1,2,3,4])
df.insert(5, 'test', category)
del df['charges']
#print(df['test'].unique())
#print(df)
X = df.drop('test', axis=1)
y = df['test']
start_time = time.time()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)
for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False, n_jobs=-1)
    clf.fit(X, y)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
print(time.time() - start_time, 's')
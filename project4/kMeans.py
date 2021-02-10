import numpy as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv("cardio_train.csv")
df = pd.DataFrame(data)
print(df.head())
sns.pairplot(df[['age', 'height', 'weight', 'active']])
plt.show()
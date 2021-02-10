import numpy as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as sm
from sklearn import datasets
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)

# Set our input x to Pressure, use [[]] to convert to 2D array suitable for model input
xlabel = "Fitness"
xlabel2 = "Age"
xlabel3 = "Income"
xlabel4 = "Gender"
xlabel5 = 'Education'
#print(df.head())
sns.pairplot(df[[xlabel, xlabel2, xlabel3, xlabel5]])
plt.show()

kmeans = cluster.KMeans(n_clusters=3, init="k-means++")
kmeans = kmeans.fit(df[[xlabel, xlabel3]])
print(kmeans.cluster_centers_)

df['Clusters'] = kmeans.labels_
print(df.head())
print(df['Clusters'].value_counts())

sns.scatterplot(x=xlabel3, y=xlabel, hue='Clusters', data=df)
plt.show()
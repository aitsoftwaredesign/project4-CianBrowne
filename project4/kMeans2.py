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
data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)
#print(df.region.unique())
xlabel = "age"
xlabel2 = "sex"
xlabel3 = "bmi"
xlabel4 = "children"
xlabel5 = "smoker"
xlabel6 = "region"
df = pd.DataFrame(data)
#print(df.head())
sns.pairplot(df[[xlabel, xlabel2, xlabel3, 'charges']])
plt.show()

kmeans = cluster.KMeans(n_clusters=5, init="k-means++")
kmeans = kmeans.fit(df[['charges', xlabel3]])
print(kmeans.cluster_centers_)

df['Clusters'] = kmeans.labels_
print(df.head())
print(df['Clusters'].value_counts())

sns.scatterplot(x=xlabel4, y='charges', hue='Clusters', data=df)
plt.show()
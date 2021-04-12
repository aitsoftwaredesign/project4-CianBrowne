import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)
del df['region']

category = pd.cut(df['charges'],bins=[0,3000,5000,10000,25000,100000],labels=[0,1,2,3,4])
df.insert(5, 'rates', category)
del df['charges']
print(df.shape)
print(df.head)

X = df.drop('rates', axis=1)
y = df['rates']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)

clf = SVC(kernel='linear', C=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(y_pred)

print(classification_report(y_test, y_pred))
plot_confusion_matrix(clf, X_test, y_test)
plt.show()
#print(X)

joblib_file = "productionModel.pkl"
joblib.dump(clf, joblib_file)
joblib_LR_model = joblib.load(joblib_file)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("insurance.csv")
df = pd.DataFrame(data)
df['smoker'].replace(['no','yes'],[0,1],inplace=True)
df['sex'].replace(['male','female'],[0,1],inplace=True)
df['region'].replace(['southwest','southeast', 'northwest', 'northeast'],[0,1,2,3],inplace=True)
print(df.region.unique())
xlabel = "age"
xlabel2 = "sex"
xlabel3 = "bmi"
xlabel4 = "children"
xlabel5 = "smoker"
xlabel6 = "region"
X = df[[xlabel6]]
y = df.charges

pre_process = PolynomialFeatures(degree=2)
# Transform our x input to 1, x and x^2
X_poly = pre_process.fit_transform(X)
# Show the transformation on the notebook
X_poly
pr_model = LinearRegression()

# Fit our preprocessed data to the polynomial regression model
pr_model.fit(X_poly, y)

# Store our predicted Humidity values in the variable y_new
y_pred = pr_model.predict(X_poly)

# Produce a scatter graph of Humidity against Pressure

plt.scatter(X, y, c = "black")
plt.xlabel("Region")
plt.ylabel("Charges")
plt.plot(X, y_pred)
plt.show()
y_new = pr_model.predict(pre_process.fit_transform([[24]]))
y_new
print("RMS Error")
print(mean_squared_error(y, y_pred))
print("Charges")
print(y_new)
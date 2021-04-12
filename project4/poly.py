import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

data = pd.read_csv("CardioGoodFitness.csv")
df = pd.DataFrame(data)
del df['Product']
df['Gender'].replace(['Male','Female'],[0,1],inplace=True)
df['MaritalStatus'].replace(['Single','Partnered'],[0,1],inplace=True)

# Set our input x to Pressure, use [[]] to convert to 2D array suitable for model input
xlabel = "Miles"
xlabel2 = "Age"
xlabel3 = "Income"
xlabel4 = "Gender"
X = df[[xlabel, xlabel2, xlabel3, xlabel4]]
y = df.Fitness

pre_process = PolynomialFeatures(degree=2)
# Transform our x input to 1, x and x^2
X_poly = pre_process.fit_transform(X)
# Show the transformation on the notebook
X_poly
pr_model = LinearRegression()

# Fit our preprocessed data to the polynomial regression model
pr_model.fit(X_poly, y)
X2 = sm.add_constant(X_poly)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

# Store our predicted Humidity values in the variable y_new
y_pred = pr_model.predict(X_poly)

# Produce a scatter graph of Humidity against Pressure


plt.plot(X, y_pred)
plt.show()
y_new = pr_model.predict(pre_process.fit_transform([[5, 80, 69000, 1]]))
y_new
print("RMS Error")
print(mean_squared_error(y, y_pred))
print("Fitness Level")
print(y_new)
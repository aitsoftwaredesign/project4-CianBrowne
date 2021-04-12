import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv("cardio_train.csv")
df = pd.DataFrame(data)


xlabel = "weight"
xlabel2 = "height"
xlabel3 = "smoke"
xlabel4 = "alco"
xlabel5 = "active"
xlabel6 = "age"
X = df[[xlabel6, xlabel, xlabel2]]
y = df.cardio

pre_process = PolynomialFeatures(degree=1)
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
plt.xlabel(xlabel)
plt.ylabel("cardio")
plt.plot(X, y_pred)
plt.show()
y_new = pr_model.predict(pre_process.fit_transform([[7655, 56.69, 171]]))
y_new
print("RMS Error")
print(mean_squared_error(y, y_pred))
print("Charges")
print(y_new)
"""
Multiple Linear Regression with Python
"""

# Remzi Alpaslan

# import library
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# import data
df = pd.read_csv("multiple_linear_regression.csv", sep=";")
# print(df)

x = df.iloc[:, [0, 2]].values
y = df.maas.values.reshape(-1, 1)

########################################################################
multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x, y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2: ", multiple_linear_regression.coef_)

print(multiple_linear_regression.predict(np.array([[10, 35], [5, 35]])))

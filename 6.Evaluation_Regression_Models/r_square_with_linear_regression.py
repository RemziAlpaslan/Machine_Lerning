"""
R-Square with Linear Regression
"""

# Remzi Alpaslan

# import library
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# import data
df = pd.read_csv("linear_regression.csv", sep=";")

# linear regression model
linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1, 1)
y = df.maas.values.reshape(-1, 1)

linear_reg.fit(x, y)

# prediction
####################################################################################################

b0 = linear_reg.predict([[0]])
# print("b0 ", b0)

b0 = linear_reg.intercept_
# print("b0 ", b0)  # y eksenini kestiği nokta intercept

b1 = linear_reg.coef_
# print("b1 ", b1)  # eğim slope

y_head = linear_reg.predict(x)

print("r_square score", r2_score(y, y_head))

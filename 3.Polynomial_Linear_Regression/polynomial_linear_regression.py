"""
Polynomial Linear Regression with Python
"""

# Remzi Alpaslan

# import library
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# import data
df = pd.read_csv("polynomial_linear_regression.csv", sep=";")
# print(df)

x = df.araba_fiyat.values.reshape(-1, 1)
y = df.araba_max_hiz.values.reshape(-1, 1)

plt.scatter(x, y)
plt.ylabel("araba_max_hiz")
plt.xlabel("araba_fiyat")
# plt.show()

# linear regression y= b0 +b1*x
# multiple linear regression y = b0 + b1*x1+b2*x2

################################################################################

lr = LinearRegression()

lr.fit(x, y)

y_head = lr.predict(x)

plt.plot(x, y_head, color="red")
# plt.show()

# print("10 milyon tl lik arabanın hız değeri: ", lr.predict([[10000]])[0][0])

###############################################################################
# polynomial linear regression y = b0 + b1*x+b2*x^2+b3*x^3+....+bn*x^n

polynomial_regression = PolynomialFeatures(degree=4)  # sayının arttırılması hata oranını düşürüyor.

x_polynomial = polynomial_regression.fit_transform(x)
print(x_polynomial)

##############################################################################
# fit
linear_regression2 = LinearRegression()
linear_regression2.fit(x_polynomial, y)

##############################################################################
y_head2 = linear_regression2.predict(x_polynomial)

plt.plot(x, y_head2, color="purple")
plt.show()

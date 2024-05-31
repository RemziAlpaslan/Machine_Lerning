"""
Random Forest Regression with Python
"""

# Remzi Alpaslan

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("random_forest_regression.csv", sep=";", header=None)
# print(df)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

###############################################################################
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)
print("7.8.seviyesinde fiyatın ne kadar olduğu: ", rf.predict([[7.8]]))

x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = rf.predict(x_)

###############################################################################
# visualize
plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("tribün level")
plt.ylabel("ücret")
plt.show()

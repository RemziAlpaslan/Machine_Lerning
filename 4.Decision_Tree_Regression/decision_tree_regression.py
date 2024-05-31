"""
Decision Tree Regression with Python
"""

# Remzi Alpaslan

# import library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# import data
df = pd.read_csv("decision_tree_regression.csv", sep=";", header=None)
# print(df)

x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

# decision tree regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x, y)

# print(tree_reg.predict([[5.5]]))
x_ = np.arange(min(x), max(x), 0.01).reshape(-1, 1)
y_head = tree_reg.predict(x_)

# virsualize

plt.scatter(x, y, color="red")
plt.plot(x_, y_head, color="blue")
plt.xlabel("tribun level")
plt.ylabel("ucret")
plt.show()

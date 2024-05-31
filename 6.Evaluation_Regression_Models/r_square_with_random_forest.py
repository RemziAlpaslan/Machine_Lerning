"""
R-Square with Random Forest Regression
"""

# Remzi Alpaslan

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("random_forest_regression.csv", sep=";", header=None)
# print(df)
x = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values.reshape(-1, 1)

###############################################################################
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x, y)

y_head = rf.predict(x)

###############################################################################
print("r_score: ", r2_score(y, y_head))

"""
Logistic Regression Easy
"""

# Remzi Alpaslan


#  libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#  read csv
data = pd.read_csv("logistic_regression.csv")
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
print(data.info())

y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

#  normalization
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# (x - min(x))/(max(x)-min(x))

#  train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))

"""
Logistic Regression Titanic
"""

# Remzi Alpaslan


#  libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#  read csv
data = pd.read_csv("train_data.csv")
print(data.info())
data.drop(["Unnamed: 0", "PassengerId"], axis=1, inplace=True)
print(data.info())

y_train = data.Survived.values
x_train = data.drop(["Survived"], axis=1)

data1 = pd.read_csv("test_data.csv")
print(data1.info())
data1.drop(["Unnamed: 0", "PassengerId"], axis=1, inplace=True)
print(data1.info())

y_test = data1.Survived.values
x_test = data1.drop(["Survived"], axis=1)



x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)
print("test accuracy {}".format(lr.score(x_test.T, y_test.T)))

"""
Model Selection
"""

# Remzi Alpaslan

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

iris = load_iris()

x = iris.data
y = iris.target

# normalization
x = (x - np.min(x) / (np.max(x) - np.min(x)))

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# knn model
knn = KNeighborsClassifier(n_neighbors=3)

# k fold CV K
accuracies = cross_val_score(estimator=knn, X=x_train, y=y_train, cv=10)

print("average accuracies: ", np.mean(accuracies))
print("average std: ", np.std(accuracies))

# test
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print("test accuracies: ", knn.score(x_test, y_test))

# ############################################################################################################################# grid search cross validation for knn

grid = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, grid, cv=10)
knn_cv.fit(x, y)

# print hyperparameter KNN algoritmasindaki K değeri

print("tuned hyperparameter K: ", knn_cv.best_params_)
print("tuned parametreye gore en iyi accuacy (best score): ", knn_cv.best_score_)

# ############################################################################################################################# Grid search CV with logistic regression

x = x[:100, :]
y = y[:100]

grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}  # l1 = lasso l2=ridge
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, grid, cv=10)
logreg_cv.fit(x, y)

print("tuned hyperparameter: (best parameters) ", logreg_cv.best_params_)
print("accuray: ",logreg_cv.best_score_)

logreg2 = LogisticRegression(C=1, penalty="l2")
logreg2.fit(x_train,y_train)
print("score:", logreg2.score(x_test,y_test))
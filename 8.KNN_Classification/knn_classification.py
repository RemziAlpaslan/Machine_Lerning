"""
KNN Classification
"""

# Remzi Alpaslan

# import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("knn_classification.csv")
# print(data.info())

data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
# print(data.tail())

M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu")
# plt.scatter(B.radius_mean, B.texture_mean, color="green", label="iyi")
# plt.xlabel("radius_mean")
# plt.ylabel("texture_mean")
# plt.legend()
# plt.show()

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# knn model
knn = KNeighborsClassifier(n_neighbors=7)  # n_neighbors = k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
# print(prediction)

print("{} knn score {}".format(7, knn.score(x_test, y_test)))

# find k value
score_list = []
for each in range(1, 15):
    knn2 = KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train, y_train)
    knn2.score(x_test, y_test)
    score_list.append(knn2.score(x_test, y_test))

plt.plot(range(1, 15), score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()

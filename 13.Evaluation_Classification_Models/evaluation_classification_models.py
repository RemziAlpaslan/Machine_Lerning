"""
Evaluation Classification Models
"""

# Remzi Alpaslan

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Import data
data = pd.read_csv("evaluation_classification_models.csv")

# Fix data
data.drop(["id", "Unnamed: 32"], axis=1, inplace=True)
# print(data.tail())

# Show data
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="kotu")
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="iyi")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
# plt.show()

# Fix data
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# Normalization data
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

# Separate data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Use method and test
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)

# print("print accuracy of rf algo:", rf.score(x_test, y_test))
# print(rf.predict([x_train[100]]))

y_pred = rf.predict(x_test)
y_true = y_test
# confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)

# cm visualization
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()

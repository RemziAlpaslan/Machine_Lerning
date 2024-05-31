"""
NLP
"""

# Remzi Alpaslan

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import nltk as nlp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# import twitter data
data = pd.read_csv(r"nlp.csv", encoding="latinı")
data = pd.concat([data.gender, data.description], axis=1)
# print(data)


data.dropna(axis=0, inplace=True)
data.gender = [1 if each == "female" else 0 for each in data.gender]
# print(data)

# cleaning data

first_description = data.description[4]
# print(first_description)
description = re.sub("[^a-zA-z]", " ", first_description)  # harf dışındaki şekilleri boşluk ile değiştir.
description = description.lower()  # büyük harfleri küçük harf ile değiştir.
# print(description)

# stop world (irrelavent words) gereksiz kelimeler.
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')

# description = description.split()
# split yerine tokenizer kullanabiliriz.

description = nltk.word_tokenize(description)
# split kullnırsak "shouldn't gibi kellimeler should ve not diye  ikiye ayrılmaz ama tokenize kullanırsak ayrılır.


# gereksiz kelimeleri çıkar.
description = [word for word in description if word not in set(stopwords.words("english"))]

# lemmatazation loved=> love  gitmek=> git
lemma = nlp.WordNetLemmatizer()
# print(description)
description = [lemma.lemmatize(word) for word in description]
# print(description)

description = " ".join(description)
# print(description)

###################################################################################################
# yapılan işlemlerin hepsini tüm data için yapıcaz.
# Cleaning Data
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-z]", " ", description)  # harf dışındaki şekilleri boşluk ile değiştir.
    description = description.lower()
    description = nltk.word_tokenize(description)
    # description = [word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
# print(description_list)

# Bag of Word
max_features = 7500
count_vectorizer = CountVectorizer(max_features=max_features, stop_words="english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()
# print("eb sık kullanılan {} kelimeler {}".format(max_features,count_vectorizer.get_feature_names_out()))

# Text Classification
y = data.iloc[:, 0].values
x = sparce_matrix

# train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
# naive bayes

nb = GaussianNB()
nb.fit(x_train, y_train)

print("accuracy: ", nb.score(x_test, y_test))

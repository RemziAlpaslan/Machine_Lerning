"""
Recommendation Systems
"""

# Remzi Alpaslan

import pandas as pd
import os

# import movie data set and look at columns
movie = pd.read_csv("movie.csv")
# print(movie.columns)

# what we need id and title
movie = movie.loc[:, ["movieId", "title"]]
# print(movie.head(10))

# import ratings data set and look at columns
rating = pd.read_csv("rating.csv")
# print(rating.columns)

# what we need ids and rating
rating = rating.loc[:, ["userId", "movieId", "rating"]]
# print(rating.head(10))

# then mege movie and rating data
data = pd.merge(movie, rating)

# now lets look at our data
print(data.head(10))

data = data.iloc[:1000000, :]

pivot_table = data.pivot_table(index=["userId"],columns=["title"],values=["rating"])
print(pivot_table.head(10))

movie_watched = pivot_table["Bad Boys (1995)"]

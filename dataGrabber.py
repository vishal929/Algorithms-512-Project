# file houses logic for grabbing necessary data, and cleaning/transforming it for use with k-nn
import pandas as pd
from math import sqrt

# grabbing iris dataset
def grabIris():
    data = pd.read_csv('iris.data',names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'], sep=',')
    return data.values

def irisDistanceFunction(elementOne, elementTwo):
    # only 4 numeric features, the last feature is a classifier label
    # will use euclidean distance
    distance = 0
    for i in range(4):
        distance += (elementOne[i] - elementTwo[i])**2
    # returning the square root for euclidean distance
    return sqrt(distance)

# grabbing our other dataset

# grabbing imdb dataset

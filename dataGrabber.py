# file houses logic for grabbing necessary data, and cleaning/transforming it for use with k-nn
import pandas as pd

# grabbing iris dataset
def grabIris():
    data = pd.read_csv('iris.data',names=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'], sep=',')
    return data

# grabbing our other dataset

# grabbing imdb dataset

# the Linear Regression class and the Boston housing data set

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine


boston = load_boston()
print(boston.data.shape)
iris = load_iris()
wine = load_wine()

from sklearn.linear_model import LinearRegression

data_x = boston.data
data_y = boston.target
model = LinearRegression()
model.fit(data_x, data_y)
print("coef")
print(model.coef_)
maxeffect = max(model.coef_)
for i in range(len(model.coef_)):
    if model.coef_[i] == maxeffect:
        print(i)
print("RM average number of rooms per dwelling has highest effection")





# K-means and the Wine data set

from sklearn.datasets import load_wine
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wine = load_wine()
for i in range(1,10):
    total = 0.0
    kmeans = KMeans(n_clusters=i, random_state=0).fit(wine.data)
    center = kmeans.cluster_centers_
    for current_lable in range(i):
        subtotal = 0
        for j in range(len(center)):
            for wine_index in range(len(wine.data)):
                if kmeans.labels_[wine_index] == j and kmeans.labels_[wine_index] == current_lable:
                    for k in range(len(center[j])):
                        subtotal += (wine.data[wine_index][k] - center[j][k]) * (wine.data[wine_index][k] - center[j][k])
        #print("lable ",current_lable," j ",j, " subtotal ",subtotal)
        total+=subtotal
    print(total)
#print(kmeans.labels_)
#print(kmeans.cluster_centers_)



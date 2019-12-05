
# K-means and the Wine data set

from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wine = load_wine()
_total = []
sse = {}
for i in range(1,10):
    total = 0.0
    kmeans = KMeans(n_clusters=i, random_state=0).fit(wine.data)
    center = kmeans.cluster_centers_
    sse[i] = kmeans.inertia_


plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

print("number of clusters on the data is 2")



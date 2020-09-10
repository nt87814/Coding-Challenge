import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('ClusterPlot.csv')

x=data.copy()
wcss=[]
for i in range(1,10):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) #the sum of squared distances of samples to their closest cluster center.

print(wcss)
plt.plot(range(1,10),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
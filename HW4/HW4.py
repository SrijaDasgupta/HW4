import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns;
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
import time
import math
sns.set()
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

def euclidean_distance(point1, point2):
    distance = 0
    for a,b in zip(point1, point2):
        distance += pow((a-b), 2)
    return math.sqrt(distance)


def cosine_similarity(point1, point2):
  A = np.array(point1)
  B = np.array(point2)
  dist = 1 - np.dot(A,B)/(np.linalg.norm(A)*np.linalg.norm(B))
  return dist

def jaccard(A, B):
    return 1 - (np.sum(np.minimum(A,B), axis = 0)/np.sum(np.maximum(A, B), axis = 0))


def calculate_centroid(cluster):
    n = len(cluster[0])
    if isinstance(cluster[0][-1], str):
        centroid = [0] * (n - 1)

        for i in range(n - 1):
            for point in cluster:
                centroid[i] += point[i]
            centroid[i] = centroid[i] / len(cluster)
    else:
        centroid = [0] * n

        for i in range(n):
            for point in cluster:
                centroid[i] += point[i]
            centroid[i] = centroid[i] / len(cluster)

    return centroid


def draw_and_scatter(clusters, centroid_centers):
    colors = ["red", "blue", "green"]
    for i, key in enumerate(clusters):
        x = []
        y = []
        cluster = clusters[key]
        for c in cluster:
            x.append(c[0])
            y.append(c[1])
        plt.scatter(x, y, marker='^', c=colors[i])

    for point in centroid_centers:
        plt.scatter(point[0], point[1], marker='s')

    plt.show()


def label_cluster(cluster):
  cl = defaultdict(int)
  for point in cluster:
    cl[point[-1]] += 1
  return cl


class KMeans:
    def __init__(self, n_clusters=10, max_iters=10, init_centroids=None, d_func=euclidean_distance, show_sse=False,
                 show_first_centroid=False, centroid_stop=True):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.init_centroids = init_centroids
        self.d_func = d_func
        self.sse_list = []
        self.show_first_centroid = show_first_centroid
        self.show_sse = show_sse

    def fit(self, data):
        start = time.time()
        if self.init_centroids is None:
            # Assign random points of data as centroids of size k (n_clusters)
            random_choice = np.random.choice(range(len(data)), self.n_clusters, replace=False)
            centroids = []

            for choice in random_choice:
                if isinstance(data[choice][-1], str):
                    centroids.append(data[choice][:-1])
                else:
                    centroids.append(data[choice])

            self.init_centroids = centroids

        for loop in range(self.max_iters):
            print("Running: ",loop)
            clusters = defaultdict(list)
            sse = 0
            # Now, assign each point to nearest centroid cluster

            for point in data:
                temp_centroid = -1
                min_dist = 99999999
                for i, centroid in enumerate(self.init_centroids):
                    if isinstance(point[-1], str):
                        d = self.d_func(point[:-1], centroid)
                    else:
                        d = self.d_func(point, centroid)
                    if d < min_dist:
                        temp_centroid = i
                        min_dist = d

                clusters[temp_centroid].append(point)

            prev_centroids = self.init_centroids.copy()
            # Now, recalculating the centroids
            for key in clusters.keys():
                cluster = clusters[key]
                self.init_centroids[key] = calculate_centroid(cluster)

            if loop == 1 and self.show_first_centroid == True:
                print("Centroids after first iteration: ", self.init_centroids)

            if self.init_centroids == prev_centroids:
                break

            for key in clusters.keys():
                cluster = clusters[key]
                ce = self.init_centroids[key]

                for p in cluster:
                    sse += euclidean_distance(ce, p)

            if self.show_sse == True and loop > 1 and self.sse_list[-1] <= sse:
                self.sse_list.pop()
                break

            self.sse_list.append(sse)

        print("Time taken:", time.time() - start)
        print("Number of iterations:", loop)
        return [self.init_centroids, clusters]



label = pd.read_csv("label.csv").to_numpy()
data = pd.read_csv("data.csv").to_numpy()
#description = pd.read_csv("data_description.txt")


arr = []

for row in range(len(data)):
  temp = []
  for col in range(len(data[row])):
    temp.append(data[row][col])
  temp.append(label[row][0])
  arr.append(temp)

arr=sorted(arr, key=lambda x: x[len(arr[0])-1], reverse=False)


target_labels = dict(label_cluster(arr))
print(target_labels)


def run(func):
    if(func is None):
        kmeans = KMeans()
    else:
        kmeans = KMeans(d_func=func)

    [centroid_centers, clusters] = kmeans.fit(arr)

    labels = {0: 0, 1: 0, 2: 0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

    for key in clusters:
      d = dict(label_cluster(clusters[key]))
      mx = 0
      s = 0
      label = ""
      for k in d:
        s += d[k]
        if d[k] > mx:
          mx = d[k]
          label = k
      labels[label] = mx

    #draw_and_scatter(clusters, centroid_centers)

    print("SSE =",kmeans.sse_list)
    print("Original Labels: ", target_labels)
    print("Predicted Labels: ", labels)

    total = 0
    mismatch = 0

    for l in target_labels:
      total += target_labels[l]
      mismatch += abs(target_labels[l] - labels[l])

    accuracy = (total - mismatch) / total

    print("Accuracy =",accuracy)



print("******* EUCLEDIAN *******")
run(None)
print("******* COSINE *******")
run(cosine_similarity)
print("******* JACCARD *******")
run(jaccard)

print("********************* PART A *********************")
print("******* EUCLEDIAN *******")
kmeans = KMeans(centroid_stop=True)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* COSINE *******")
kmeans = KMeans(centroid_stop=True, d_func=cosine_similarity)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* JACCARD *******")
kmeans = KMeans(centroid_stop=True, d_func=jaccard)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)


print("********************* PART B *********************")
print("******* EUCLEDIAN *******")
kmeans = KMeans(show_sse=True)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* COSINE *******")
kmeans = KMeans(show_sse=True, d_func=cosine_similarity)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* JACCARD *******")
kmeans = KMeans(show_sse=True, d_func=jaccard)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)

print("********************* PART C *********************")
print("******* EUCLEDIAN *******")
kmeans = KMeans(max_iters=100, show_sse=False, centroid_stop=False)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* COSINE *******")
kmeans = KMeans(max_iters=100, show_sse=False, centroid_stop=False, d_func=cosine_similarity)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)
print("******* JACCARD *******")
kmeans = KMeans(max_iters=100, show_sse=False, centroid_stop=False, d_func=jaccard)
[centroid_centers, clusters] = kmeans.fit(arr)
print(kmeans.sse_list)




import numpy as np
import os


def create_centroidds():
    centroids = []
    centroids.append([ 5.0, 0.0])
    centroids.append([ 5.0, 70.0])
    centroids.append([50.0, 90.0])
    return np.array(centroids)

def print_label_data(cluster_label, new_centroids):
    print('Result of k-Means clustering:')
    for data in cluster_label:
        print("data: {}, cluster label:{}".format(data[0], data[1]))
    print("Last centroids: {}".format(new_centroids))

def compute_distance(item, centroid):
    tmp = (item-centroid)
    tmp = tmp**2
    tmp = np.sum(tmp)
    tmp = np.sqrt(tmp)
    return tmp

def find_label(distance):
    idx_min = min(distance, key=distance.get)
    return idx_min

def gen_new_center(item, centroid):
    return np.array(item+centroid)/2

def train_k_means(data, centroid, epochs):
    cluster_label = []
    size = len(data)
    K = len(centroids)

    for epoch in range(epochs):
        #print('epoch-{}, centroid:{}'.format(epoch, centroid))
        for i in range(0, size):
            distance = {}
            for j in range(0, K):
                distance[j] = compute_distance(data[i], centroids[j])
            label_idx = find_label(distance)
            centroids[label_idx] = gen_new_center(data[i], centroids[label_idx])
            if epoch == epochs-1:
                cluster_label.append([data[i], label_idx])
    return cluster_label, centroids




if __name__ == '__main__':
    filename = "/Users/aodandan/repos/acode/k-means/data.csv"
    data = np.genfromtxt(filename, delimiter=',')
    centroids = create_centroidds()
    total_iteration = 2

    #print(compute_distance(np.array([5.0, 0.0]), np.array([0.0, 1.0])))
    #print(find_label({0:1, 1:2, 2:0.1, 3:1.5}))
    #print(gen_new_center(np.array([5.0, 0.0]), np.array([0.0, 1.0])))
    cluster_label, new_centroids = train_k_means(data, centroids, total_iteration)

    print_label_data(cluster_label, new_centroids)




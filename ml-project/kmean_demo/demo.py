# Kmean 实现

import numpy as np
from sklearn.datasets import *
import matplotlib.pyplot as plt

def kmeanDIY(data, k):
    #将数据集转换为numpy数组
    samples = np.array(data)
    n_samples = samples.shape[0]   #点数


    point_selected = np.random.choice(n_samples,k,replace=False)  #从数据中随机选K个点，并且不重复
    centroids = samples[point_selected]
    #标记变量，作为跳出循环条件，当一次循环中没有一个点的分类被改变，则终止循环
    print(type(centroids[0]),centroids)
    cluster_changed = True
    cluster_assment = np.zeros([n_samples, 2])  # 第一列记录当前点归属那一类，第二列距离中心点的距离
    colors = ['r', 'g', 'k', 'c', 'y', 'tan', 'gold', 'steelblue', 'darkred']
    # print(samples.shape[0])
    while (cluster_changed):
        cluster_changed = False
        # 循环每一个点
        for i in range(n_samples):
            dis_vec = []
            min_dist = np.inf
            min_index = 0
            # for i2 in range(k):     #循环计算每个中心点的距离，并比较距离，取最小的距离，作为分类点
            #     distance = sum(np.square(samples[i] - centroids[i2]))
            #     if distance < min_dist :
            #         min_dist = distance
            #         min_index = i2

            dis_vec.extend(np.square(samples[i] - centroids).sum(axis = 1))
            min_dist = np.array(dis_vec).min()
            min_index = dis_vec.index(min_dist)

            # 判断点的距离中心是否改变
            if cluster_assment[i,0] != min_index :
                cluster_assment[i,:] = min_index,min_dist
                cluster_changed = True

        # 更新聚类中心
        for i in range(k):
            tmp = samples[cluster_assment[:,0] == i,:]
            centroids[i,:] = np.mean(tmp, axis = 0)    #把每一类中的点的均值作为新的聚类点

        # for i in range(n_clusters):
        #     tmp = cluster_assment[:, 0] == i
        #     tmp = X[tmp, :]
        #     plt.scatter(tmp[:, 0], tmp[:, 1], c=colors[i], s=50)
        # plt.show()
    return centroids, cluster_assment

def get_distance(point_cluster,data_set):
    point_cluster = np.array(point_cluster)
    return np.square(point_cluster - data_set).sum(axis=1)


def kmean_adv(data_set,k):
    samples = np.array(data_set)
    num_samples = samples.shape[0]

    point_selected = np.random.choice(num_samples,1,replace=False)
    cluster_assment = np.zeros([num_samples, 2])  # 第一列记录当前点归属那一类，第二列距离中心点的距离
    cluster_changed = True
    centroids = np.zeros((k,2))
    centroids[0]=(samples[point_selected])
    diff_dis = get_distance(centroids[0],data_set)
    max_dis = np.array(diff_dis).max()
    max_index = list(diff_dis).index(max_dis)
    centroids[1]=(samples[max_index])
    if k > 2 :
        for i in range(2,k):
            max_diff = 0
            max_index = 0
            for j in range(len(centroids)):
                dis_points = get_distance(centroids[j],data_set)
                dis_index = np.argsort(dis_points)
                dis_diff = dis_points[dis_index[1]]
                if dis_diff > max_diff :
                    max_diff = dis_diff
                    max_index = dis_index[1]
            centroids[i]=(samples[max_index])
    while (cluster_changed):
        cluster_changed = False
        # 循环每一个点
        for i in range(num_samples):
            dis_vec = []
            min_dist = np.inf
            min_index = 0
            for i2 in range(k):     #循环计算每个中心点的距离，并比较距离，取最小的距离，作为分类点
                distance = sum(np.square(samples[i] - centroids[i2]))
                if distance < min_dist :
                    min_dist = distance
                    min_index = i2

            # dis_vec.extend(np.square(samples[i] - centroids).sum(axis = 1))
            # min_dist = np.array(dis_vec).min()
            # min_index = dis_vec.index(min_dist)

            # 判断点的距离中心是否改变
            if cluster_assment[i,0] != min_index :
                cluster_assment[i,:] = min_index,min_dist
                cluster_changed = True

        # 更新聚类中心
        for i in range(k):
            tmp = samples[cluster_assment[:,0] == i,:]
            centroids[i,:] = np.mean(tmp, axis = 0)    #把每一类中的点的均值作为新的聚类点

        # for i in range(n_clusters):
        #     tmp = cluster_assment[:, 0] == i
        #     tmp = X[tmp, :]
        #     plt.scatter(tmp[:, 0], tmp[:, 1], c=colors[i], s=50)
        # plt.show()
    return centroids, cluster_assment




X,y = make_blobs(n_samples=350,n_features=2,centers=3,cluster_std=0.5,random_state=0)
n_clusters = 3
centroids, cluster_assment = kmean_adv(X,n_clusters)
# print(centroids)

# colors = ['r','g','k','c','y','tan','gold','steelblue','darkred']
# for i in range(n_clusters):
#     tmp = cluster_assment[:,0] == i
#     tmp = X[tmp,:]
#     plt.scatter(tmp[:,0],tmp[:,1],c = colors[i],s=50)
# plt.show()

loss = []
for i in range(2,7):
    _,cluster_assment = kmean_adv(X,i)
    loss.append(np.sum(cluster_assment[:,1]))
plt.plot(loss)
plt.show()
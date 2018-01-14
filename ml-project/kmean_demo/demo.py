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
    return np.square(data_set - point_cluster).sum(axis=1)


# 想法错误：kmean++为先计算每个点与聚类点的距离，取小值，然后再取所有点中距离的最大值。
# 这里先计算了一个聚类点到全部点的距离取最小，再取最大，这样就会导致第二个和第三个聚类点相距很近
def kmean_adv(data_set,k):
    samples = np.array(data_set)
    num_samples = samples.shape[0]

    point_selected = np.random.choice(num_samples,1,replace=False)
    cluster_assment = np.zeros([num_samples, 2])  # 第一列记录当前点归属那一类，第二列距离中心点的距离
    cluster_changed = True
    centroids = np.zeros((2,2))
    centroids[0]=(samples[point_selected])
    diff_dis = get_distance(centroids[0],data_set)
    max_dis_index = np.array(diff_dis).argsort()[::-1]
    print(diff_dis[max_dis_index[0]] == np.array(diff_dis).max())
    print(diff_dis[max_dis_index[:10]])
    max_index = max_dis_index[0]
    centroids[1]=(samples[max_index])
    if k > 2 :
        for i in range(2,k):
            max_diff = 0
            max_index = 0
            for j in range(len(centroids)):
                dis_points = get_distance(centroids[j],data_set)
                dis_index = np.argsort(dis_points)
                # if dis_points[dis_index[0]] == 0:
                #     dis_diff = dis_points[dis_index[1]]
                #     max_index = dis_index[0]
                # else:
                #     dis_diff = dis_points[dis_index[0]]
                dis_diff = dis_points[dis_index[1]]
                # print(dis_points[dis_index[:10]])
                # print(dis_diff)
                if dis_diff > max_diff :
                    max_diff = dis_diff
                    max_index = dis_index[1]

            # print(max_index)
            centroids = np.insert(centroids,i,values=samples[max_index],axis=0)
            # print(centroids)
        # color_s = ['b', 'y', 'steelblue']
        # for i in range(len(centroids)):
        #     plt.scatter(centroids[i][0], centroids[i][1], c=color_s[i], s=90, marker='<')
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
        # color_s = ['b', 'y', 'steelblue']
        # for i in range(len(centroids)):
        #     plt.scatter(centroids[i][0], centroids[i][1], c=color_s[i], s=60, marker='<')
        # plt.show()
    return centroids, cluster_assment


def kmean2_base(data_set,k):
    samples = np.array(data_set)
    n_samples = samples.shape[0]
    cluster_changed = True
    cluster_assment = np.zeros([n_samples, 2])  # 第一列记录当前点归属那一类，第二列距离中心点的距离

    point_selected = np.random.choice(n_samples,1)
    centroids = samples[point_selected]
    distance = (samples - centroids[0]).sum(axis = 1)
    max_dis = np.array(distance).max()
    max_index = list(distance).index(max_dis)
    centroids = np.insert(centroids,1,values=samples[max_index],axis = 0)
    if k > 2 :
        for i in range(2,k):
            list_dis = []
            max_index = 0
            max_dis = 0
            for j in range(n_samples):
                distance = (centroids - samples[j]).sum(axis = 1)
                if max_dis < distance.min():
                    max_dis = distance.min()
                    max_index = j
            centroids = np.insert(centroids, i, values=samples[max_index], axis=0)
    color_s = ['b', 'y', 'steelblue']
    for i in range(len(centroids)):
        plt.scatter(centroids[i][0], centroids[i][1], c=color_s[i], s=90, marker='<')
    while (cluster_changed):
        cluster_changed = False
        # 循环每一个点
        for i in range(n_samples):
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
    return centroids, cluster_assment




colors = ['r','g','k','c','y','tan','gold','steelblue','darkred']

X,y = make_blobs(n_samples=150,n_features=2,centers=3,cluster_std=0.5,random_state=0)
n_clusters = 3
centroids, cluster_assment = kmean2_base(X,n_clusters)
# print(centroids)

for i in range(n_clusters):
    tmp = cluster_assment[:,0] == i
    tmp = X[tmp,:]
    plt.scatter(tmp[:,0],tmp[:,1],c = colors[i],s=50)

# color_s = ['steelblue','y','b']
# print(centroids)
# for i in range(len(centroids)):
#     print(i)
#     plt.scatter(centroids[i][0], centroids[i][1], c=color_s[i], s=60,marker='<')
plt.show()

# loss = []
# for i in range(2,8):
#     _,cluster_assment = kmean_adv(X,i)
#     loss.append(np.sum(cluster_assment[:,1]))
# plt.plot(loss)
plt.show()
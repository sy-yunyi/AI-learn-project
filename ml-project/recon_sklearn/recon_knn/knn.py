import numpy as np
import pandas as pd
import operator
from sklearn import *

def classify(in_x,data_set,labels,k):
    """
    kNN分类器
    :param in_x:  目标特征向量
    :param data_set: 数据集
    :param labels: 分类向量
    :param k: k值
    :return: None
    """
    diffMat=data_set-in_x          #计算输入向量与数据集每一个向量的差值
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5        #计算欧式距离
    sq_sort=distances.argsort()      #获取距离排序下标
    classCount = {}
    for i in range(k):
        votel = labels[sq_sort[i]]             #获取排在前K位的特征值
        classCount[votel]=classCount.get(votel,0)+1    #统计特征出现的次数，并存储在一个字典中
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   #排序
    return sortedClassCount[0][0]

    # data_set_size=data_set.shape[0]
    # #计算距离(4 lines)
    # diff_mat=np.tile(in_x,(data_set_size,1))-data_set #点相减
    # sq_diff_mat=diff_mat**2             #点差求平方
    # sq_distance=sq_diff_mat.sum(axis=1)  #距离平方求和
    # distance = sq_distance**0.5      #开根号
    # sorted_dist_indices=distance.argsord()
    # classCount={}
    # for i in range(k):
    #     #获取前K个特征值
    #     vote_i_label=labels[sorted_dist_indices[i]]
    #     #当键不在字典中的时候，取默认值为0
    #     classCount[vote_i_label]=classCount.get(vote_i_label,0)+1
    # sorted_class_count=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    # #
    # return sorted_class_count[0][0]

def auto_norm(data_set):
    """
    归一化数据：将任意取值范围内的特征转化为0-1区间的值
    公式：new_value=(current_val-min)/(max-min)
    :param data_set: 数据集
    :return:
    """
    min_vals=data_set.min(0)  #获取数组内最小值
    max_vals=data_set.max(0)  #获取数组内最小值
    ranges=max_vals-min_vals  #最大值与最小值差

    m = data_set.shape[0]   #size:行数
    norm_data_set = data_set - np.tile(min_vals,(m,1))
    norm_data_set=norm_data_set/np.tile(ranges,(m,1))   #(old_cla
    return norm_data_set,ranges,min_vals

def classifer_sk(in_x,data_set,labels,k):
    scaler ,norm_data_set = norm_data(data_set)
    norm_in_x = scaler.transform(in_x)
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(norm_data_set,labels)
    result_class = clf.predict(norm_in_x)
    return result_class

def norm_data(data_set):
    scaler = preprocessing.MinMaxScaler()
    norm_data_set = scaler.fit_transform(data_set)
    return scaler,norm_data_set
if __name__ == '__main__':
    pass
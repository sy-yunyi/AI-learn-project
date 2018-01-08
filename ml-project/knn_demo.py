import numpy as np

import operator

def create_data_set():
    """
    返回测试数据集和分类向量
    :return:np_data:测试数据集；np_lables :分类向量
    """
    groups=[[0,0],[1.0,1.0],[0,0.1],[1.0,1.1]]
    np_data=np.array(groups)
    lables=['A','B','A','B']
    np_lables=np.array(lables)
    return np_data,np_lables



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

if __name__ == '__main__':
    data_groups,labels=create_data_set()

    d=np.array([1.0,1.0],[3,4],[6,4],[0.9,0.8])

    target_class=classify(d,data_groups,labels,2)
    print(target_class)
    # print(data_groups-d)
    # print((data_groups-d)**2)
    # print(data_groups)
    # print(labels)
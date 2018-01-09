import numpy as np
import pandas as pd
import operator


def create_data_set():
    """
    创建数据集和目标向量
    :return: np_data_array:数据集的numpy数组、
    :return:np_lables: 目标向量的numpy数组
    """
    # data_groups=[[1.0,0.9],[1,2],[4,5],[6,4]]
    data_groups = [[0, 0], [1.0, 1.0], [0, 0.1], [1.0, 1.1]]
    # lables=['A','A','B','B']
    lables = ['A', 'B', 'A', 'B']
    np_data_array=np.array(data_groups)
    np_lables=np.array(lables)
    return np_data_array,np_lables

def classify(in_x,data_set,lables,k):
    """
    kNN分类器实现：

    :param in_x: 目标特征向量
    :param data_set: 特征数据集
    :param lables: 分类向量
    :param k: k值
    :return: None
    """
    #获取特征数据集的行数-------目的：为了创建相减的数组
    data_set_size=data_set.shape[0]
    #目标特征向量与特征数据集相减
    #np.tile(A，rep)----重复A的各个维度 ,rep:重复次数，可以为元组eg:(2,3),(2,3,3)
    #tile([1,2],(2,2))---->[1,2] => [[1,2] , [1,2]] => [[1,2,1,2] , [1,2,1,2]]
    diffMat=np.tile(in_x,(data_set_size,1))-data_set
    #求差的平方
    sq_DiffMat=diffMat**2
    #将每行的相加   axis=1:行
    sq_distances=sq_DiffMat.sum(axis=1)
    #求根号
    distances=sq_distances**0.5
    #将计算结果排序。
    #argsort()返回将数据排序后数据在原始数组中的下标
    sort_index=distances.argsort()
    #创建一个字典，来存放前K个
    classCount={}

    for i in range(k):
        votelabel=lables[sort_index[i]]
        classCount[votelabel]=classCount.get(votelabel,0)+1
    sorted_ClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sorted_ClassCount[0][0]



if __name__ == '__main__':
    data_set,lables=create_data_set()
    test_in=[1,1]
    print(classify(test_in,data_set,lables,2))
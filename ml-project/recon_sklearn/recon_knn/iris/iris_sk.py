import numpy as np
from recon_sklearn.recon_knn.knn import classifer_sk
import pandas as pd


def load_to_data_set(file_path):
    """
    加载数据，并返回数据集，目标分类
    返回的数据格式化为Numpy数组
    :param file_path: 文件路径
    :return: np_data_set 数据集
    :return: np_data_lables 目标分类
    :return: count_class_per  每类的数量
    """
    #加载文件，并赋给新的列名
    data_set_pd=pd.read_csv(file_path,names=['sepal_length','sepal_width','petal_length','petal_width','iris_name'])
    data_set=[data[:4] for data in data_set_pd.values ]   #将读取的数据为前四列放至特征向量数据集
    count_class_per=data_set_pd.groupby("iris_name").size()
    # print(count_class_per)
    data_lables=[data[-1] for data in data_set_pd.values]  #将读取的数据的最后一列保存至目标分类标签向量
    np_data_set=np.array(data_set)
    np_data_lables=np.array(data_lables)                   #转换为numpy数组
    # print(data_lables)
    return np_data_set,np_data_lables,count_class_per                      #返回数据集和目标分类

def get_data_ratio(data_set,ratio,count_class,labels):
    """
    在数据集中去留出一定比例做测试数据集，
    返回训练数据集和测试数据集及对应的目标分类向量
    :param data_set: 数据集
    :param ratio: 训练集比例
    :param count_class: 每个分类的数量
    :param labels: 目标分类向量
    :return: data_train_new  训练数据集
    :return: data_test_new   留出的测试集
    :return:labels_train_new  训练目标分类向量
    :return:labels_test_new   测试集目标分类向量
    """
    count_class_train=[int(count*ratio) for count in count_class]       #获得训练集中每个分类的数量
    count = [0,count_class[0]]                      #记录每个分类开始下标
    for i in range(len(count_class)-1):
        count.append(count[i+1]+count_class[i])
    data = []                   #临时保存每个分类的数据
    labels_per=[]               #临时保存每个分类的目标分类的分类向量
    data_test_new=[]            #测试数据集
    data_train_new=[]           #训练数据集
    labels_train_new=[]         #训练分类向量
    labels_test_new=[]          #测试分类向量
    for i in range(len(count_class)):
        data.append(data_set[count[i]:count[i+1]])              #将数据集按分类分割
        labels_per.append(labels[count[i]:count[i+1]])          #将分类向量按分类分割
        data_train_new.extend(data[i][0:count_class_train[i]])          #按比例取数据集中的特征向量，并放去新的训练集中
        data_test_new.extend(data[i][count_class_train[i]:count_class[i]])      #按比例去数据集中的特征向量，并放去新的测试集中
        labels_train_new.extend(labels_per[i][0:count_class_train[i]])          #按比例取训练集对应分类向量
        labels_test_new.extend(labels_per[i][count_class_train[i]:count_class[i]])      #按比例取测试集对应分类向量
    print('测试集大小为: %d,训练集大小为: %d ' %(len(data_test_new),len(data_train_new)))
    # print(len(labels_train_new),len(data_train_new))
    return data_train_new,data_test_new,labels_train_new,labels_test_new

def classifier_iris_sk_test():
    ratio = 0.5
    data_set, lables,count_class = load_to_data_set('../data/iris.data')    #加载数据文件
    data_set,data_set_test,lables,lables_test=get_data_ratio(data_set,ratio,count_class,lables)  #获取训练数据集，测试数据集及对应分类向量

    result_calss = classifer_sk(data_set_test,data_set,lables,3)

    error_count = 0.0
    for i in range(len(result_calss)):
        if result_calss[i] != lables_test[i]:
            error_count += 1
    print('错误率为：%0.2f%% '%(float(error_count)/len(data_set_test)*100))

if __name__ == '__main__':
    classifier_iris_sk_test()
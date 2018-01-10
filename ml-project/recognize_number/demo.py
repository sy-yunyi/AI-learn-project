from knn import classify
import numpy as np
import os
import time
import random


def get_random_data(data_number_per,ratio):
    """
    随机获得数据集一定比例的数据，并返回数据集及分类
    :param data_number_per: 每个分类中所包含数据
    :param ratio: 要获得的比例
    :return: data_per_set 随机获得的数据集
    :return: lables  目标分类标签
    """
    # print(len(data_number_per[0]))
    data_per_set = []    #随机取得的数据集
    lables=[]            #分类标签
    for i in range(10):     #遍历所有分类
        n = int(len(data_number_per[i])*ratio)      #获取每个分类中要获取的数量
        while n>0:
            rn=random.randint(0, len(data_number_per[i])-1)     #随机获取一个0-数据集个数的整数
            data_per_set.append(data_number_per[i][rn])         #将取得是一个特征值加入返回的数据集
            del data_number_per[i][rn]                          #将加入的特征删除，防止出现重复
            n-=1
            lables.extend([i])                                  #重新获得标签
    # print(len(data_per_set),len(lables))
    return data_per_set,lables

def file2Matrix(file_path,ratio=1):
    """
    从文件夹中读取全部文件，并转化为数据集
    :param file_path: 文件夹路径
    :return: returnVector:数据集
    :return:returnLabels:分类向量
    """
    file_labels=[]
    fd=os.walk(file_path)                   #读取文件夹内所有文件
    for root,dirs,files in fd:              #遍历获得的文件名列表
        line_datas=[]                       #保存全部特征向量
        file_labels=[]
        data_number_per=[[],[],[],[],[],[],[],[],[],[]]
        for file in files:
            fr=open(file_path+'/'+file)         #打开文件
            file_labels+=file.split('_')[0]     #将文件名切分，获得目标特征标签
            line_data=[]                        #保存每个文件的内容
            file_data_lines=fr.readlines()      #读取每个文件的全部行
            for line in file_data_lines:        #遍历所有行
                line=line.strip()               #去除每行的前后空格
                line_data+=line                 #将每个文件中每一行加在一起，放在一个list中
            line_datas.append(line_data)
            data_number_per[int(file.split('_')[0])].append(line_data)
            # print(len(data_number_per[0]))
        line_datas,file_labels = get_random_data(data_number_per,ratio)     #调用获取一定比例数据集
        # print(len(data_number_per[0]))
        returnLabels=np.array(file_labels,dtype=np.int)                     #将数据集格式转换为numpy数组
        returnVector=np.array(line_datas,dtype=np.int)
        return returnVector,returnLabels



def classify_number_test(data_set_path,test_data_path,k,ratio=1):
    """
    应用knn识别数字，并计算正确率
    :param data_set_path: 数据集
    :param test_data_path: 测试集
    :param k: k值
    :return: None
    """
    start = time.clock()              #设置时间点
    data_set, lables = file2Matrix(data_set_path,ratio)             #读取训练数据集
    data_set_test, lables_test = file2Matrix(test_data_path)        #读取测试数据集
    true_count = 0
    # print(lables_test.dtype)

    for i in range(data_set_test.shape[0]):
        result_lable = classify(data_set_test[i], data_set, lables, k)          #调用kNN分类器返回分类结果
        # print('分类器返回结果为：%d,真实结果为：%d' % (result_lable, lables_test[i]))
        if result_lable == lables_test[i]:
            true_count = true_count + 1
    end = time.clock()
    print('k值为：%d,正确率为：%0.2f%%,耗时：%0.6f' % (k,(true_count / data_set_test.shape[0] * 100),(end - start)))

def test_diffK(data_set_path, test_data_path,ratio=1):
    """
    测试不同的K值
    :param data_set_path: 训练数据集
    :param test_data_path: 测试数据集
    :param ratio: 训练数据集比例
    :return: None
    """
    for i in range(5):
        classify_number_test(data_set_path, test_data_path, i+1, 1)


if __name__ == '__main__':

    # starttime = datetime.datetime.now()

    data_set_path='../data/digits/training_digits'
    test_data_path='../data/digits/test_digits'
    test_diffK(data_set_path, test_data_path,1)
    # classify_number_test(data_set_path, test_data_path, 1,1)
    # data_set, lables = file2Matrix(data_set_path)

    # endtime = datetime.datetime.now()


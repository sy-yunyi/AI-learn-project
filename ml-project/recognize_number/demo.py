from knn import classify
import numpy as np
import os
import time
import random


def get_random_data(data_number_per,ratio):
    # print(len(data_number_per[0]))
    data_set_ran=[]
    data_per_set = []
    lables=[]
    for i in range(10):
        n = int(len(data_number_per[i])*ratio)
        while n>0:
            rn=random.randint(0, len(data_number_per[i])-1)
            data_per_set.append(data_number_per[i][rn])
            del data_number_per[i][rn]
            n-=1
            lables.extend([i])
    print(len(data_per_set),len(lables))
    return data_per_set,lables

def file2Matrix(file_path,ratio=1):
    """
    从文件夹中读取全部文件，并转化为数据集
    :param file_path: 文件夹路径
    :return: returnVector:数据集
    :return:returnLabels:分类向量
    """
    file_labels=[]
    fd=os.walk(file_path)
    for root,dirs,files in fd:
        line_datas=[]
        file_labels=[]
        data_number_per=[[],[],[],[],[],[],[],[],[],[]]
        for file in files:
            fr=open(file_path+'/'+file)
            file_labels+=file.split('_')[0]
            line_data=[]
            file_data_lines=fr.readlines()
            for line in file_data_lines:
                line=line.strip()
                line_data+=line
            line_datas.append(line_data)
            data_number_per[int(file.split('_')[0])].append(line_data)
            # print(len(data_number_per[0]))
        line_datas,file_labels = get_random_data(data_number_per,ratio)
        # print(len(data_number_per[0]))
        returnLabels=np.array(file_labels,dtype=np.int)
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
    start = time.clock()
    data_set, lables = file2Matrix(data_set_path,ratio)
    data_set_test, lables_test = file2Matrix(test_data_path)
    error = 0
    # print(lables_test.dtype)

    for i in range(data_set_test.shape[0]):
        result_lable = classify(data_set_test[i], data_set, lables, k)
        # print('分类器返回结果为：%d,真实结果为：%d' % (result_lable, lables_test[i]))
        if result_lable == lables_test[i]:
            error = error + 1
    end = time.clock()
    print('k值为：%d,正确率为：%0.2f%%,耗时：%0.6f' % (k,(error / data_set_test.shape[0] * 100),(end - start)))


if __name__ == '__main__':

    # starttime = datetime.datetime.now()

    data_set_path='../data/digits/training_digits'
    test_data_path='../data/digits/test_digits'

    classify_number_test(data_set_path, test_data_path, 1,1)
    # data_set, lables = file2Matrix(data_set_path)

    # endtime = datetime.datetime.now()


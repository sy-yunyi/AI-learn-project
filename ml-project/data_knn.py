import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform
import operator



def file2matrix(filename):
    """
    将文本记录转换到numpy数组的解析程序
    :param filename: 文件名
    :return: np_array_data 特征向量
    :return：return_trait   目标变量向量
    """
    fr=open(filename)
    dataSet=fr.readlines()
    numberLines=len(dataSet)
    np_array_data=np.zeros((numberLines,3))
    return_trait=[]
    index=0
    for line in dataSet:
        no_n_data=line.strip()
        data_list=no_n_data.split('\t')
        np_array_data[index,:]=data_list[0:3]
        return_trait.append(int(data_list[-1]))
        index=index+1
    return np_array_data,return_trait

def file_to_matrix(filename):
    """
    将文本记录转换到numpy数组的解析程序
    :param filename: 数据集文件名
    :return: return_mat ：特征值向量
    :return: class_label_vector :目标变量向量
    """
    fr=open(filename)       #打开并加载文件
    array_lines=fr.readlines()   #以行读取文件
    number_of_lines=len(array_lines)      #得到文件行数

    return_mat=np.zeros((number_of_lines,3))  #返回一个新的填充为0的矩阵
    class_label_vector=[]
    index=0
    for line in array_lines:
        line=line.strip()                   #去除行的空格
        list_from_line=line.split('\t')    #将行进行切片，间隔符为‘\t’
        return_mat[index:]=list_from_line[0:3]      #将前3列的数据赋给返回矩阵
        class_label_vector.append(int(list_from_line[-1]))    #将最后一列赋给目标向量
        index=index+1
    return return_mat,class_label_vector


def create_matplotlab_img(data_set,labels):
    """
    创建散点图展示数据并分析
    :param data_set: 特征数据
    :param labels: 分类向量
    :return: None
    """

    #设置系统字体
    if platform.system()=='Windows':
        zh_font = font_manager.FontProperties() #zh_font = font_manager.FontProperties('字体路径')

    #初始化数据
    type_1_x = []
    type_1_y = []
    type_2_x = []
    type_2_y = []
    type_3_x = []
    type_3_y = []

    for i in range(len(labels)):
        if labels[i]==1:
            type_1_x.append(data_set[i][0])
            type_1_y.append(data_set[i][1])
        if labels[i]==2:
            type_2_x.append(data_set[i][0])
            type_2_y.append(data_set[i][1])
        if labels[i]==3:
            type_3_x.append(data_set[i][0])
            type_3_y.append(data_set[i][1])

    fig = plt.figure()
    ax=fig.add_subplot(111)
    #设置数据属性
    type_1=ax.scatter(type_1_x,type_1_y,s=20,c='red',alpha=0.9,marker='*')
    type_2=ax.scatter(type_2_x,type_2_y,s=40,c='blue',fontproperties=zh_font)
    type_3=ax.scatter(type_3_x,type_3_y,s=60,c='green')

    plt.title('约会对象数据分析')
    plt.ylabel('the time of plaied game ')
    plt.xlabel('每周消耗的冰淇淋公升数')
    ax.legend((type_1,type_2,type_3),('dislike','normal','like'),loc='upper left')
    plt.show()

def auto_norm1(data_set):
    """
    归一化数据
    :param data_set:
    :return:
    """
    max=data_set.max()             #获取数组中最大值
    min=data_set.min()              #获取数组中最小值
    diffMaxMin=max-min                 #最大与最小差值
    min_mat=np.zeros(np.shape(data_set))   #创建一个与数据集相同大小的0数组
    min_mat[...]=min                        #将最小值赋给最小值数组
    print(min_mat)
    diff_mat=data_set-min_mat              #数据集减去最小值

    norm_value=diff_mat/diffMaxMin       #归一化：数据集减去最小值除去最大与最小的差值

    return norm_value

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
    # print(distances)
    sq_sort=distances.argsort()      #获取距离排序下标
    classCount = {}
    for i in range(k):
        votel = labels[sq_sort[i]]             #获取排在前K位的特征值
        classCount[votel]=classCount.get(votel,0)+1    #统计特征出现的次数，并存储在一个字典中
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)   #排序
    return sortedClassCount[0][0]


def dating_class_test():
    """
    应用测试集测试分类机的错误率
    :return: None
    """
    hold_out_ratio = 0.10  #拿出作为测试集的数据比例
    data_set,labels = file_to_matrix('./data/dating_test_set_2.txt')
    norm_data_set,ranges,min_vals=auto_norm(data_set)
    size=norm_data_set.shape[0]  #获得数据集行数
    num_test_size = int(size * hold_out_ratio)   #保留行数
    error_count = 0.0  #错误统计
    for i in range(num_test_size):
        classifier_result=classify(norm_data_set[i,:],norm_data_set[num_test_size:size],labels[num_test_size:size],5)
        print('分类器返回：%d, 真实答案为：%d'% (classifier_result,labels[i]))
        if classifier_result!= labels[i]:
            error_count+=1.0
        print('分类器错误率为：%0.2f%%' % (error_count/(float(num_test_size))*100))

if __name__ == '__main__':
    # data_mat,class_label_vector=file_to_matrix('./data/dating_test_set_2.txt')
    # auto_norm1(data_mat)
    dating_class_test()
    # create_matplotlab_img(data_mat,class_label_vector)

    # print(mat)

    # print(vector)
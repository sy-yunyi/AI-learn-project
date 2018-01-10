import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt

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

def classify_iris(in_x,data_set,lables,k):
    """
    kNN分类器，实现对iris的分类
    利用欧式距离公式，计算待分类向量与数据集中所有向量的距离，取最接近的k个
    返回最接近的一个分类
    :param in_x: 目标特征向量
    :param data_set: 数据集
    :param lables: 目标分类
    :param k: k值
    :return: 分类结果
    """
    m = len(data_set)                       #获取数据集大小
    in_set=np.tile(in_x,(m,1))              #将待分类向量格式化为与数据集行数相同
    diff_mat=data_set-in_set                #求数据集与待分类向量的差
    sq_diff_mat=diff_mat**2                 #求平方
    sum_diff_mat=sq_diff_mat.sum(axis=1)    #求和
    distances=sum_diff_mat**0.5                 #求根号，计算出距离
    sorted_distances=np.argsort(sum_diff_mat)     #将获得距离进行排序,返回的为排序好的数据在原始列表中的下标
    classCount = {}
    for i in range(k):
        class_one=lables[sorted_distances[i]]
        classCount[class_one]=classCount.get(class_one,0)+1    #将获得的分类保存至一个字典中，并统计数量
    sorted_class=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)        #将获得分类字典排序,返回为一个元素为tuple的列表
    return sorted_class[0][0]



def auto_norm(data_set):
    """
    数据归一化，将数据归一都0-1或-1到1，使得所有数据具有相同的比重
    :param data_set: 数据集
    :return: norm_data 归一化后的数据集
    """
    min = data_set.min(axis=0)      #求矩阵中的每列最小值
    max = data_set.max(axis=0)      #求矩阵中的每列最小值
    ranges = max - min              #最大值与最小值的差
    m = data_set.shape[0]           #或者去数据集的行数
    min_mat = np.tile(min,(m,1))    #使的最小值的行数与数据集相同
    norm_data=(data_set-min_mat)/np.tile(ranges,(m,1))       #归一化：(current_val-min)/(max-min)
    return norm_data


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




def creat_matplotlib_img(data_set,labels):

    # 初始化数据
    type_1_x = []
    type_1_y = []
    type_2_x = []
    type_2_y = []
    type_3_x = []
    type_3_y = []

    for i in range(len(labels)):
        if labels[i] == 'Iris-setosa':
            type_1_x.append(data_set[i][0])
            type_1_y.append(data_set[i][2])
        if labels[i] == 'Iris-versicolor':
            type_2_x.append(data_set[i][0])
            type_2_y.append(data_set[i][2])
        if labels[i] == 'Iris-virginica':
            type_3_x.append(data_set[i][0])
            type_3_y.append(data_set[i][2])

    fig=plt.figure()
    ax=fig.add_subplot(111)

    # 设置数据属性
    type_1 = ax.scatter(type_1_x, type_1_y, s=20, c='red', alpha=0.9, marker='*')
    type_2 = ax.scatter(type_2_x, type_2_y, s=40, c='blue',marker='<')
    type_3 = ax.scatter(type_3_x, type_3_y, s=60, c='green')

    plt.title('iris种类分析')
    plt.ylabel('sepal length')
    plt.xlabel('sepal width')
    ax.legend((type_1, type_2, type_3), ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'), loc='upper left')
    plt.show()


def iris_classifer_test():
    """
    应用测试集测试分类机的错误率
    :return:None
    """
    ratio = 0.5             #训练集比例
    data_set, lables,count_class = load_to_data_set('./data/iris.data')    #加载数据文件

    # creat_matplotlib_img(data_set,lables)                             #绘画matplotlib散点图

    data_set = auto_norm(data_set)                                  #归一化
    data_set,data_set_test,lables,lables_test=get_data_ratio(data_set,ratio,count_class,lables)  #获取训练数据集，测试数据集及对应分类向量
    # data_set_test=data_set[int(len(data_set)*ratio):]
    # lables_test = lables[int(len(data_set)*ratio):]
    error_count=0.0                      #错误统计
    for i in range(len(data_set_test)):
        result_class = classify_iris(data_set_test[i], data_set, lables, 3)
        if result_class != lables_test[i]:
            print("出错了。分类器返回去为：%s,真实结果为：%s." % (result_class,lables_test[i]))
            error_count += 1
    print("错误率为:%0.2f%%" % (error_count/len(data_set_test)*100))

if __name__ == '__main__':
    iris_classifer_test()
    # in_x=[4.9,3.0,1.4,0.2]

    # print(result_class)
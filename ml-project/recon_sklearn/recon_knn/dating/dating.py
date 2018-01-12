# 使用kNN算法优化单身狗约会
# 使用留出法
# 使用matplotlib 绘制图像
# 打印分类器错误率

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import platform
from knn import classify
from knn import auto_norm

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

def dating_class_test():
    """
    应用测试集测试分类机的错误率
    :return: None
    """
    hold_out_ratio = 0.10  #拿出作为测试集的数据比例
    data_set,labels = file_to_matrix('../data/dating_test_set_2.txt')
    norm_data_set,ranges,min_vals=auto_norm(data_set)
    size=norm_data_set.shape[0]  #获得数据集行数
    num_test_size = int(size * hold_out_ratio)   #保留行数
    print(num_test_size)
    error_count = 0.0  #错误统计
    for i in range(num_test_size):
        classifier_result=classify(norm_data_set[i,:],norm_data_set[num_test_size:size],labels[num_test_size:size],5)

        if classifier_result!= labels[i]:
            print('分类器返回：%d, 真实答案为：%d，下标为：%d' % (classifier_result, labels[i],i))
            error_count+=1.0
    print('分类器错误率为：%0.2f%%' % (error_count/(float(num_test_size))*100))

def classify_gui(k):
    """
    :param k: k值
    :return:人群分类
    """
    data_mat, class_label_vector = file_to_matrix('../data/dating_test_set_2.txt')
    fly_distances = float(input("请输入飞行里程数："))
    icecream = float(input("请输入消耗冰淇淋公升数："))
    play_time = float(input("请输入玩游戏花费时间百分比："))
    norm_data_set, ranges, min_vals = auto_norm(data_mat)
    data_person=np.array([fly_distances,icecream,play_time])
    norm_person_data=(data_person-min_vals)/ranges
    class_person=['不喜欢','一般','极具魅力']
    label_person=classify(norm_person_data,norm_data_set,class_label_vector,k)
    return class_person[label_person-1]


def classify_person():
    """
    对给定的数据进行人群分类判断
    :return:
    """
    #定义人群分类：[0，1，2]
    ff_miles=float(input("每年飞行常客里程数："))
    ice_cream=float(input('每周消耗的冰淇淋公升数：'))
    percent_game=float(input('玩游戏所消耗的时间百分比：'))
    data_mat,class_label_vector=file_to_matrix('../data/dating_test_set_2.txt')
    norm_dating_data_set,ranges,min_vals=auto_norm(data_mat)
    in_x=np.array([ff_miles,ice_cream,percent_game])   #待验证数据

    norm_in_x=(in_x - min_vals)/(ranges)
    classify_result=classify(norm_in_x,norm_dating_data_set,class_label_vector,3)
    return classify_result

if __name__ == '__main__':
    # dating_class_test()
    dating_class_test()
    # print(classify_gui(5))
    # auto_norm1(data_mat)
    # dating_class_test()
    # create_matplotlab_img(data_mat,class_label_vector)
    # classify_person()
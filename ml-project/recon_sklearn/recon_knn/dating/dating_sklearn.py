from sklearn import neighbors
from sklearn import preprocessing
import numpy as np
from recon_sklearn.recon_knn.knn import classifer_sk,norm_data

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


def sklearn_classifier_test():
    data_set, lables =file_to_matrix('../data/dating_test_set_2.txt')
    # scaler ,norm_data_set = norm_data(data_set)
    hold_out_ratio = 0.1
    size = data_set.shape[0]
    num_test_size = int(size * hold_out_ratio)  # 保留行数
    error_count = 0.0  # 错误统计
    result_class = classifer_sk(data_set[:num_test_size],data_set[num_test_size:size],lables[num_test_size:size],5)
    # print(num_test_size)
    for i in range(num_test_size):
        if result_class[i] != lables[i]:
            print('分类器返回：%d, 真实答案为：%d，下标为：%d' % (result_class[i], lables[i], i))
            error_count += 1
    print('错误率为：%0.2f%% '% (error_count/float(num_test_size) * 100))

def classify_person():
    """
    :return:
    """
    ff_miles = float(input("每年飞行常客里程数："))
    ice_cream = float(input('每周消耗的冰淇淋公升数：'))
    percent_game = float(input('玩游戏所消耗的时间百分比：'))
    data_set, lables = file_to_matrix('../data/dating_test_set_2.txt')
    in_x=np.array([ff_miles,ice_cream,percent_game]).reshape(1,3)   #待验证数据
    class_person = ['一般，不用去。','魅力一般，可以去。','魅力十足，必须去！']
    result_class = classifer_sk(in_x,data_set,lables,3)
    print(class_person[result_class[0]-1])

if __name__ == '__main__':
    sklearn_classifier_test()
    # classify_person()



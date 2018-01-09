#kNN算法：手写数字识别系统

import numpy as np
from knn import classify
from os import listdir

def image_to_vector(filename):
    """

    :param filename:
    :return:
    """
    fr=open(filename)
    lines=fr.readlines()
    image_data=[]
    for line in lines:
        line=line.strip()
        image_data+=line
    return image_data

def img_to_vector(filename):
    """
    将图像转换为向量
    :param filename: 目标图像的文件名
    :return: return_vect  返回数据向量
    """
    return_vect=np.zeros((1,1024))  #创建1*（32*32）数据向量
    fr=open(filename)
    for i in range(32):
        line_str=fr.readline() #第i行
        for j in range(32):
            return_vect[0,32*0+j]=int(line_str[j])
    return return_vect

def file_to_matrix(file_path):
    """

    :param file_path:
    :return:
    """
    img_list=listdir(file_path)
    lables_img=[]
    return_vectore=np.zeros((len(img_list),1024))
    index=0
    for img_path in img_list:
        img_path.strip()
        img_lable=img_path.split('_')
        lables_img.extend(img_lable)
        data_vector=img_to_vector(file_path+'/'+img_path)
        return_vectore[index:]=data_vector
    return_vectore.dtype=np.int
    return return_vectore,lables_img

def hand_writing_class_test1(data_file_path,test_file_path):
    """

    :param data_file_path:
    :param test_file_path:
    :return:
    """
    data_set,lables_img=file_to_matrix(data_file_path)
    data_set_test,lables_img_test = file_to_matrix(test_file_path)


    # print(img_list)


def hand_writing_class_test():
    """
    测试识别手写数字分类正确率
    :return: None
    """

    #第一步，创建训练集内容
    hw_labels=[]
    training_file_list = listdir('../data/digits/training_digits')  #获取目录内容
    m=len(training_file_list)
    training_data_set = np.zeros((m,1024))  #利用行数创建训练集集合
    for i in range(m):
        filename_str = training_file_list[i]
        # file_str = filename_str.split('.')[0]  #将文件名截取
        # class_num_str = int(file_str.split('_')[0])
        class_num_str=int(filename_str.split('_')[0])
        hw_labels.append(class_num_str)  #将分类添加至分类向量

        img_vector = img_to_vector('../data/digits/training_digits/%s' % filename_str)
        training_data_set[i,:]=img_vector
        # print(training_data_set[i])
    #第二步，创建测试集
    test_file_list=listdir('../data/digits/test_digits')
    m_test = len(test_file_list)
    error_count = 0.0
    for i in range(m_test):
        filename_str = test_file_list[i]

        class_num_str=int(filename_str.split('_')[0])
        # print(class_num_str)
        test_img_vector=img_to_vector('../data/digits/test_digits/%s' % filename_str)
        # print((training_data_set[i]-test_img_vector).sum())
        # print(class_num_str)
        #创建一个5NN分类模型
        classifier_result=classify(test_img_vector,training_data_set,hw_labels,5)
        print(classifier_result)
        # if classifier_result != class_num_str:
        #     error_count+=1.0
        #     print('出错了，%d'%class_num_str)
    # print('错误率为：%0.2%%' %(error_count/m_test*100))


if __name__ == '__main__':
    # image_to_vector('../data/digits/test_digits/0_0.txt')
    # file_to_matrix('../data/digits/training_digits')
    # file_to_matrix('../data/digits/training_digits')
    hand_writing_class_test()


import numpy as np
import pandas as pd
from math import log

def load_data_set():
    """
    my notes for this function
    获取测试数据集
    :return:postingList
    :return:class_list
    """
    fd=open('./data/post.txt')
    lines=fd.readlines()

    class_list=[]
    postingList=[]
    for line in lines:
        line=line.strip()
        line_list=line.split(' ')
        postingList.append(line_list[0:len(line_list)-1])
        class_list.append(line_list[-1])
    np_class = np.array(class_list,dtype=np.int)
    np_post = np.array(postingList)

    return postingList,np_class

def create_vocab_list(data_set):
    """
    创建一个包含在所有文档中出现且不重复的词汇表
    :param data_set:待处理文本数据集
    :return:不重复的词汇表
    """
    # vocabSet=set([])        #创建一个空集
    # for i_vec in data_set:
    #     vocabSet = vocabSet | set(i_vec)     #求并集
    # return list(vocabSet)

    data_list=[]
    for vec in data_set:
        data_list.extend(vec)
    data=set(data_list)
    return(list(data))

def setOf2WordVec(in_word,vocabList):
    recVector=[0]*len(vocabList)
    # print(recVector)
    for word in in_word:
        if word in vocabList:
            recVector[vocabList.index(word)]=1
        else:
            print ('%s 不在词汇表中' % word)
    return recVector

def set_of_words_to_vec(vocab_list,inupt_set):
    """
    将目标文档转化为向量：词集模型（set of words model)
    :param vocab_list: 词汇表
    :param inupt_set: 待处理文档
    :return: 转化为的文档向量
    """

    return_vec=len(vocab_list)*[0]   #创建一个所含元素都为0的列表
    for word in inupt_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print('单词：%s 并没有在当前的词汇表中' % word)
    return return_vec

def train_nb1(train_data_set,train_category):
    train_size = len(train_data_set)
    num_words = len(train_data_set[0])
    p_abusive= sum(train_category)/float(train_size)
    p_0_num = np.ones(num_words)
    p_1_num = np.ones(num_words)
    p0Demon = 2.0
    p1Demon = 2.0
    index_i=0
    for i_vec in train_data_set:
        if train_category[index_i] == 0:
            p_0_num += i_vec
            p0Demon += sum(train_data_set[index_i])
        elif (train_category[index_i] == 1 ):
            p_1_num += i_vec
            p1Demon += sum(train_data_set[index_i])
        index_i +=1
    p1Vector = p_1_num/p1Demon
    p0Vector = p_0_num/p0Demon
    return p1Vector,p0Vector,p_abusive

def train_nb(train_data_set,train_category):
    """
    训练朴素贝叶斯算法模型
    :param train_data_set: 训练集
    :param train_category: 分类向量
    :return:
    """
    num_train_docs = len(train_data_set)
    num_words = len(train_data_set[0])
    p_abusive = sum(train_category)/float(num_train_docs)   #标注侮辱性文档的概率

    p_0_num = np.ones(num_words)    #正常概率向量
    p_1_num = np.ones(num_words)      #侮辱概率向量

    p_0_denom = 2.0
    p_1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 0:   #正常文档
            p_0_num += train_data_set[i]
            p_0_denom += sum(train_data_set[i])
        else:
            p_1_num += train_data_set[i]
            p_1_denom += sum(train_data_set[i])
    p_0_vect = np.log(p_0_num/p_0_denom)
    p_1_vect = np.log(p_1_num / p_1_denom)
    print(p_0_vect,p_1_vect,p_abusive)
    return p_0_vect,p_1_vect,p_abusive

def classify_nb1(vec_to_classify,p_0_v,p_1_v,p_class_0):
    p_vec_0 = vec_to_classify * p_0_v + log (p_class_0)
    p_vec_1 = vec_to_classify * p_1_v + log(1.0 - p_class_0)
    p0 = sum(p_vec_0)
    p1 = sum(p_vec_1)
    print(p0,p1)

def classify_nb(vec_to_classify,p_0_vect,p_1_vect,p_class):
    """
    朴素贝叶斯分类器
    :param vec_to_classify:待分类向量
    :param p_0_vect: p(0)
    :param p_1_vect: p(1)
    :param p_class: 类别概率
    :return:
    """

    p_0 = sum(vec_to_classify * p_0_vect) + np.log(p_class)
    p_1 = sum(vec_to_classify * p_1_vect) + np.log(1.0 - p_class)
    if p_0 > p_1:
        return 0
    else:
        return 1

def testing_nb():
    """

    :return:
    """
    data_list,class_list = load_data_set()
    my_vocab_list = create_vocab_list(data_list)
    train_data_set = []
    for post_doc in data_list:
        train_data_set.append(set_of_words_to_vec(my_vocab_list,post_doc))
    p_0_v,p_1_v,p_class = train_nb(train_data_set,list_classes)

    classify_nb(p_0_v,p_1_v,p_class)



if __name__ == '__main__':
    # posting_list,class_vet=load_data_set()
    # vocabList=create_vocab_list(posting_list)
    # word_vec=setOf2WordVec(['I', 'need', 'help'],vocabList)

    list_posts,list_classes=load_data_set()
    my_vocab_list = create_vocab_list(list_posts)
    training_data_set = []
    for post_doc in list_posts:
        training_data_set.append(set_of_words_to_vec(my_vocab_list,post_doc))  #将文本转换成向量后放入数据集

    p_0_v, p_1_v, p_class_0 = train_nb(training_data_set,list_classes)  #朴素贝叶斯模型
    classify_nb(training_data_set[1],p_0_v, p_1_v, p_class_0)
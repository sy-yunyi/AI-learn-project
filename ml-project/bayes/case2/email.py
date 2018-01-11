import numpy as np
import re
import os
import random



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

def bag_of_words_to_vec(vocab_list,inupt_set):
    """
    将目标文档转化为向量：词袋模型（bag of words model)
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

def classify_nb(vec_to_classify,p_0_vect,p_1_vect,p_class):
    """
    朴素贝叶斯分类器
    :param vec_to_classify:待分类向量
    :param p_0_vect: p(0)
    :param p_1_vect: p(1)
    :param p_class: 类别概率
    :return: 分类
    """

    p_0 = sum(vec_to_classify * p_0_vect) + np.log(p_class)
    p_1 = sum(vec_to_classify * p_1_vect) + np.log(1.0 - p_class)

    if p_0 > p_1:
        return 0
    else:
        return 1

def text_parse(words):
    """
    文本处理起
    :param words:待处理文本
    :return: 处理完成文本词汇列表
    """
    # words = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    words_list = [word for word in re.compile('\\W*').split(words) if len(word) > 0]
    return words_list

def loat_text():
    file_path = './email/spam'
    for root,dirs,file_list in os.walk(file_path):
        for file in file_list:
            fd = open(file_path+'/'+file)
            data_file = fd.read()
            words_list = text_parse(data_file)
            print(words_list)


def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1,26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        class_list.append(1)
        full_text.extend(word_list)

        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    # print(doc_list,len(doc_list))
    vocab_list = create_vocab_list(doc_list)            #创建词汇表
    training_set = list(range(50))
    test_set = []
    #构建测试集容器
    for i in range(10):
        rand_index = int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    training_mat = []
    training_classes = []
    for doc_index in training_set:
        training_mat.append(set_of_words_to_vec(vocab_list,doc_list[doc_index]))
        training_classes.append(class_list[doc_index])

    p_0_v,p_1_v,p_spam = train_nb(np.array(training_mat),np.array(training_classes))

    for doc_index in test_set:
        word_vector = set_of_words_to_vec(vocab_list,doc_list[doc_index])
        if classify_nb(np.array(word_vector),p_0_v,p_1_v,p_spam) != class_list[doc_index]:
            print()
        else:
            print ()
if __name__ == '__main__':
    # word = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    # # print(word.split())
    # p=re.compile('\\W*').split(word)
    # print(p)
    # text_parse()
    # loat_text()
    spam_test()
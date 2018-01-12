# 检测恶意评论
# 数据较少，仅用了很少的几句话
# 没有在找数据集，使用原本的数据进行了验证

from recon_sklearn.recon_bayes.bayes import *

import numpy as np


def load_data_set():
    """
    my notes for this function
    获取测试数据集
    :return:postingList
    :return:class_list
    """
    fd=open('../data/post.txt')
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

    # classify_nb(【测试数据】,p_0_v,p_1_v,p_class)



if __name__ == '__main__':
    # posting_list,class_vet=load_data_set()
    # vocabList=create_vocab_list(posting_list)
    # word_vec=setOf2WordVec(['I', 'need', 'help'],vocabList)

    list_posts,list_classes=load_data_set()
    my_vocab_list = create_vocab_list(list_posts)
    training_data_set = []
    for post_doc in list_posts:
        training_data_set.append(set_of_words_to_vec(my_vocab_list,post_doc))  #将文本转换成向量后放入数据集
    # p_0_v, p_1_v, p_class_0 = train_nb(training_data_set,list_classes)  #朴素贝叶斯模型

    clf = classifier_nb_sk(training_data_set,list_classes)
    commont = ['非恶意评论','恶意评论']

    # result_class = classify_nb(training_data_set[0],p_0_v, p_1_v, p_class_0)

    result_class = clf.predict([training_data_set[0]])

    print(commont[result_class[0]])
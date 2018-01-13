# 使用RSS源信息，分析地区差异
# 最后输出使用贝叶斯算法进行分类的错误率，并且打印不同地区出现频率最高的一组词
import random
import feedparser
import numpy as np
import re
import operator
from recon_sklearn.recon_bayes.bayes import *

def calc_most_freq(vocab_list,full_text,most_count = 30):
    """
    计算出现频率最多的词汇
    :param vocab_list: 不重复词汇表
    :param full_text: 文本全部词汇表
    :param most_count: 最多移除数量
    :return: 移除目标高频词汇后的词表
    """
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_text.count(token)
    sorted_freq = sorted(freq_dict.items(),key=operator.itemgetter(1),reverse=True)
    # print(sorted_freq)
    return sorted_freq[:most_count]

def get_stop_words():
    file_path = './data/stop_words_en.txt'
    fd = open(file_path)
    file_data = fd.read()
    words_list = text_parse(file_data)
    return set(words_list)
    # words_list_set = set(words_list)
    # doc_list_set = set(doc_list)
    # del_stop_words_list = doc_list_set - words_list_set
    # print(len(del_stop_words_list),len(doc_list_set))


def local_words(feed_0, feed_1):
    """
    :param feed_0:
    :param feed_1:
    :return:
    """
    min_len = min(len(feed_0['entries']),len(feed_1['entries']))

    doc_list = []
    class_list = []
    full_text = []

    for i in range(min_len):
        word_list = text_parse(feed_0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

        word_list = text_parse(feed_1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)

    vocab_list = create_vocab_list(doc_list)

    #去除出现频率最高的一组数据
    # top_words = calc_most_freq(vocab_list,full_text,20)
    # for pair_w in top_words:
    #     if pair_w[0] in vocab_list:
    #         vocab_list.remove(pair_w[0])

    training_set = list(range(2 * min_len))
    test_set = []

    for i in range(20):
        rand_index = int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    training_mat = []
    train_classes = []
    #生成训练集

    for doc_index in training_set:
        training_mat.append(bag_of_words_to_vec(vocab_list,doc_list[doc_index]))
        train_classes.append(class_list[doc_index])

    # p_0_v ,p_1_v,p_class = train_nb(np.array(training_mat),np.array(train_classes))  #训练模型
    # print(p_0_v,p_1_v)
    clf = classifier_nb_sk(training_mat,train_classes)
    score = clf.score(training_mat,train_classes)
    print(score)
    # error_count = 0.0
    # p_0_v = []
    # p_1_v = []
    # for doc_index in test_set:
    #     word_vector = bag_of_words_to_vec(vocab_list,doc_list[doc_index])
    #     p_class=clf.predict([np.array(word_vector)])
    #     p_v_i = clf.predict_log_proba([np.array(word_vector)])
    #     # print(p_v_i[0])
    #     p_0_v.extend(p_v_i[0])
    #     p_1_v.extend(p_v_i[0])
    #     # print(p_1_v_i)
    #     # if classify_nb(np.array(word_vector),p_0_v,p_1_v,p_class) != class_list[doc_index]:
    #     if p_class != class_list[doc_index]:
    #         error_count += 1
    # print('错误率为：%0.2f%%' % (float(error_count)/len(test_set)*100))
    # print(len(p_0_v))
    # print(p_1_v)
    # return vocab_list,p_0_v,p_1_v

def get_top_words(ny,sf):
    """
    获取最高频率使用单词
    :param ny: 词汇源
    :param sf: 词汇源
    :return: None
    """
    vocab_list , p_0_v,p_1_v = local_words(ny,sf)
    top_ny = []
    top_sf = []
    for i in range(len(p_0_v)):
        if p_0_v[i] >-5.0:
            top_ny.append((vocab_list[i],p_0_v[i]))
        if p_1_v[i] >-5.0:
            top_sf.append((vocab_list[i],p_1_v[i]))
    sorted_ny = sorted(top_ny,key= lambda x:x[1],reverse=True)
    sorted_sf = sorted(top_sf,key= lambda x:x[1],reverse=True)

    print('NY**'*20)
    for item in sorted_ny:
        print (item)
    print('SF**'*20)
    for item in sorted_sf:
        print(item)

if __name__ == '__main__':
    ny_rss = 'https://newyork.craigslist.org/search/stp?format=rss'
    sf_rss = 'https://sfbay.craigslist.org/search/stp?format=rss'
    feed_0 = feedparser.parse(ny_rss)
    feed_1 = feedparser.parse(sf_rss)
    local_words(feed_0,feed_1)
    # print(tc_rss.keys())
    # get_top_words(feed_0,feed_1)
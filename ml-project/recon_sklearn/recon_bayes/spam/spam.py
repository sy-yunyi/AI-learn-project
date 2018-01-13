# 过滤垃圾邮件
# 输出分类器错误率
# 使用数据集为固定25封垃圾邮件25封正常邮件
# 使用留出法，在原始数据集中随机选取一定比例数据进行验证

from recon_sklearn.recon_bayes.bayes import *

def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    for i in range(1,26):
        word_list = text_parse(open('../data/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        class_list.append(1)
        full_text.extend(word_list)

        word_list = text_parse(open('../data/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)            #创建词汇表
    training_set = list(range(50))
    test_set = []
    #构建测试集容器
    for i in range(30):
        rand_index = int(random.uniform(0,len(training_set)))
        test_set.append(training_set[rand_index])
        del (training_set[rand_index])

    training_mat = []
    training_classes = []
    for doc_index in training_set:
        training_mat.append(set_of_words_to_vec(vocab_list,doc_list[doc_index]))
        training_classes.append(class_list[doc_index])

    # p_0_v,p_1_v,p_spam = train_nb(np.array(training_mat),np.array(training_classes))
    clf = classifier_nb_sk(training_mat, training_classes)
    error_count = 0.0
    for doc_index in test_set:
        word_vector = set_of_words_to_vec(vocab_list,doc_list[doc_index])
        # if classify_nb(np.array(word_vector),p_0_v,p_1_v,p_spam) != class_list[doc_index]:
        if clf.predict([np.array(word_vector)]) != class_list[doc_index]:
            error_count += 1
            # print()
        # else:
            # print ()
    print('错误率为：%0.2f%%.' % (error_count/float(len(test_set))*100))




if __name__ == '__main__':
    # word = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
    # # print(word.split())
    # p=re.compile('\\W*').split(word)
    # print(p)
    # text_parse()
    # loat_text()
    spam_test()
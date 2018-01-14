import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载文件
def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('./data/testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))

    return data_mat, label_mat

# 阶跃函数


def sigmoid(in_x):
    return 1.0/(1+np.exp(- in_x))


# 梯度上升函数
def grad_ascent(data_mat_in, class_labels):
    data_mat = np.mat(data_mat_in)
    labels_mat = np.mat(class_labels).transpose()
    m, n = np.shape(data_mat)
    alpha = 0.001
    max_cycles = 500
    weights = np.ones((n,1))
    for i in range(max_cycles):
        h = sigmoid(data_mat * weights)
        error = (labels_mat - h)
        weights = weights + alpha * data_mat.transpose()*error
    return weights

# 画出数据集和logistic回归最佳拟合直线的函数
def plot_best_fit(wei):
    weights = wei.getA()
    data_mat ,labels_mat = load_data_set()
    data_arr = np.array(data_mat)
    n = np.shape(data_arr)[0]
    x_cord1 = [] ; y_cord1 = []
    x_cord2 = [] ; y_cord2 = []
    for i in range(n):
        if int(labels_mat[i]) == 1:
            x_cord1.append(data_arr[i,1])
            y_cord1.append(data_arr[i,2])
        else:
            x_cord2.append(data_arr[i,1])
            y_cord2.append(data_arr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1,y_cord1,s = 30 ,c= 'r', marker= 's')
    ax.scatter(x_cord2, y_cord2, s = 30, c = 'g')
    x = np.arange(-3.0,3.0,0.1)

    y = (-weights[0] - weights[1] * x ) /weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

# 随机梯度上升
def stoc_grad_ascent(data_mat, class_labels):

    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights

# 改进的随机梯度上升算法

def stoc_grad_ascent1(data_mat, class_labels, num_iter = 150):
    m, n = np.shape(data_mat)
    weights = np.ones(n)
    for i in range(num_iter):
        data_index = np.arange(m).tolist()
        # print(data_index)
        for j in range(m):
            alpha = 4/(1.0+j+i) + 0.01
            rand_index = int(np.random.uniform(0,len(data_index)))
            h = sigmoid(sum(data_mat[rand_index] * weights))
            error = class_labels[rand_index] - h
            # print(rand_index)
            weights = weights + alpha * error * data_mat[rand_index]
            del(data_index[rand_index])
    return weights


def classify_vecter(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5 :
        return 1.0
    else:
        return 0.0

def colic_test():
    fr_train = open('./data/horseColicTraining.txt')
    fr_test = open('./data/horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split()
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stoc_grad_ascent1(np.array(training_set),training_labels,500)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vecter(np.array(line_arr),train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vec*100)
    print('the error rate of the test is : %0.2f%%' % error_rate)
    return error_rate

def multi_Test():
    num_test =10
    error_sum = 0.0
    for i in range(num_test):
        error_sum += colic_test()
    print('after %d iterations the average error rate is :%0.2f%%' %(num_test,error_sum/float(num_test)))




if __name__ == '__main__':
    data_mat_in, class_labels = load_data_set()
    w = grad_ascent(data_mat_in, class_labels)
    # plot_best_fit(w)
    multi_Test()
    # print(w)
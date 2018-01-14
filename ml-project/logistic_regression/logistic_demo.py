import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def stoc_grad_ascent(data_mat, class_labels):

    m, n = np.shape(data_mat)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(data_mat[i] * weights))
        error = class_labels[i] - h
        weights = weights + alpha * error * data_mat[i]
    return weights



if __name__ == '__main__':
    data_mat_in, class_labels = load_data_set()
    w = grad_ascent(data_mat_in, class_labels)
    plot_best_fit(w)
    # print(w)
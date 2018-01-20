import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
data = np.loadtxt('wine.txt')

def pca_diy(data_set,n):
    """

    :param data_set: 数据集
    :param n: 降维后保存信息相对于原始数据的百分比
    :return:选中的特征向量
    """

    mean_v = np.mean(data_set,axis=0)
    data_new  = (data_set - mean_v)/np.std(data_set,axis=0)
    cov_data = np.cov(data_new, rowvar= 0)
    eig_vals, eig_vects = np.linalg.eig(cov_data)
    eig_index = np.argsort(eig_vals)[::-1]
    eig_vals_v =([sum(eig_vals[eig_index[:i]]) for i in range(len(eig_vals))] /sum(eig_vals))
    eig_vals_n = np.argwhere(eig_vals_v > n)
    print(eig_vals_n)
    tmp = eig_index[:eig_vals_n[0][0]]
    eig_vec_selected = eig_vects[:, tmp]
    print(tmp)
    # pdb.set_trace()

    return eig_vec_selected


def task2(data_set,n):
    """
    :param data_set: 数据集
    :param n: 降维后保存信息相对于原始数据的百分比
    :return:选中的特征向量
    """
    mean_v = np.mean(data_set,axis=0)
    # data_new  = (data_set - mean_v)/np.std(data_set,axis=0)
    data_new  = (data_set - mean_v)

    cov_data = np.cov(data_new, rowvar= 0)
    eig_vals, eig_vects = np.linalg.eig(cov_data)
    eig_index = np.argsort(eig_vals)[::-1]
    eig_vals_v = [sum(eig_vals[eig_index[:i]]) for i in range(len(eig_vals))] /sum(eig_vals)
    eig_vals_n = np.argwhere(eig_vals_v > n)
    tmp = eig_index[:eig_vals_n[0][0]]
    eig_vec_selected = eig_vects[:, eig_index[:2]]
    # pdb.set_trace()

    return eig_vec_selected, len(tmp)


def draw_matplotlib_img(data_mat):
    x = data_mat[:,0]
    y = data_mat[:,1]
    # for i in range(len(data_mat)):
    # x.append(data_mat[i][0])
    # y.append(data_mat[i][1])
    # pdb.set_trace()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=20, c='red', alpha=0.9, marker='*')
    plt.show()



if __name__ == '__main__':

    vectors, n = task2(data,0.6)
    print(n)
    data_new = np.dot(data, vectors)
    # print(data_new)
    # print(n)
    # vectors = pca_diy(data,0.6)
    # data_new = np.dot(data, vectors)
    draw_matplotlib_img(data_new)
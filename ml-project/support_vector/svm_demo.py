import numpy as np
import pandas as pd

def load_data_set(file_name):
    data_mat = []
    labels_class = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]),float(line_arr[1])])
        labels_class.append(float(line_arr[2]))
    return data_mat,labels_class
def select_j_rand(i,m):
    j = i
    while(j == i):
        j = int(np.random.uniform(0,m))
    return j

def clip_alpha(aj,h,l):
    if aj > h :
        aj = h
    if l > aj :
        aj = l
    return aj
def smo_simple(data_mat_in, class_labels, c, toler, max_iter):
    data_mat = np.mat(data_mat_in)
    labels_mat = np.mat(class_labels).transpose()
    b = 0
    m, n = np.shape(data_mat)
    alphas = np.mat(np.zeros((m,1)))
    iter = 0
    while(iter < max_iter):
        alpha_pairs_changed = 0
        for i in range(m):
            fxi = float(np.multiply(alphas,labels_mat).T * (data_mat * data_mat[i,:].T))+b
            ei = fxi - float(labels_mat[i])




if __name__ == '__main__':
    pass
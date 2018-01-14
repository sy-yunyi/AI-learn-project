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
    



import numpy as np
import pandas as pd
from math import log
import operator
import matplotlib.pyplot as plt


def create_data_set():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# 计算制定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntrits = len(dataSet)
    classCount = {}
    for i_vec in dataSet:
        class_lable=i_vec[-1]
        classCount[class_lable]=classCount.get(class_lable,0)+1
    shanonEnt=0.0
    for key in classCount:
        prob = float(classCount[key])/numEntrits
        shanonEnt -= prob * log(prob,2)
    return shanonEnt

#按照给定特征值划分数据集
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for i_vec in dataSet:
        if i_vec[axis] == value:
            reduceFeatVec=i_vec[:axis]
            reduceFeatVec.extend(i_vec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

# 选择最好的数据集划分
def chooseDataFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)

    bestIntGain = 0         #
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVal = set(featList)
        newEntropy = 0.0
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestIntGain):
            bestFeature = i
            bestIntGain = infoGain
    return bestFeature

# 返回出现次数最多的类别
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        class_count[vote] = class_count.get(vote,0)+1
    sorter_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sorter_class_count[0][0]

# 创建树   labels 改变了
def create_tree(data_set,labels):
    labels_class = labels[:]
    class_list = [example[-1] for example in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data_set[0]) ==1:
        return majority_cnt(class_list)
    best_feat = chooseDataFeatureToSplit(data_set)
    best_feat_label = labels[best_feat]
    my_tree = {best_feat_label:{}}
    del (labels[best_feat])
    feat_values = [example[best_feat] for example in data_set]
    unique_vals = set(feat_values)
    for value in unique_vals:
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(splitDataSet(data_set,best_feat,value),sub_labels)
    return my_tree

# 使用文本注解绘制树节点
# 定义文本框和箭头格式
decision_node = dict(boxstyle = 'sawtooth' , fc = '0.8')
leaf_node = dict(boxstyle = 'round4' , fc = '0.8')
arrow_args = dict(arrowstyle = '<-')

#绘制带箭头的注解
def plot_node(node_text , center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_text,xy = parent_pt,xycoords = 'axes fraction' ,xytext = center_pt, ha = 'center' ,bbox = node_type,arrowprops = arrow_args)

def create_plot():
    fig = plt.figure(1,facecolor = 'white')
    fig.clf()
    create_plot.ax1 = plt.subplot(111,frameon = False)
    plot_node('决策节点',(0.5,0.1),(0.1,0.5),decision_node)
    plot_node('叶节点',(0.8,0.1),(0.3,0.8),leaf_node)
    plt.show()



# 使用分类的分类函数
def classify_tree(input_tree,feat_labels,test_vec):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.keys():
        if test_vec[feat_index] ==key :
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify_tree(second_dict[key],feat_labels,test_vec)
            else:
                class_label = second_dict[key]
    return class_label


if __name__ == '__main__':
    data_set ,labels =create_data_set()
    class_labels = labels.copy()
    my_tree = create_tree(data_set,labels)
    result_class = classify_tree(my_tree,class_labels,[1,0])
    print(result_class)


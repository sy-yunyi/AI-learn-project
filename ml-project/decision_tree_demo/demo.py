import numpy as np
import pandas as pd


def create_data_set():
    data_set=[]
    lables=[]
    return data_set,lables

#计算制定数据集的香农熵
def calcShannonEnt(dataSet):
    numEntrits=len(dataSet)
    classCount={}
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

#选择最好的数据集划分
def chooseDataFeatureToSplit(dataSet):


